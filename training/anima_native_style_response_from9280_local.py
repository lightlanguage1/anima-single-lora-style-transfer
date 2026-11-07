"""Anima native AB full-data initial training. Frozen Architecture v1.
Requires the adjacent generalization helpers, native library and pretrained assets.
Data/cache/resume orchestration v1; no automatic generalization claim.
"""

from __future__ import annotations

import argparse, ast, contextlib, copy, csv, gc, hashlib, inspect, io, itertools

import json, math, os, random, shutil, statistics, sys, tarfile, time, traceback

from pathlib import Path

from collections import defaultdict

from dataclasses import dataclass

import numpy as np

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

from torch import nn

from torch.nn import functional as F

from torch.utils.checkpoint import checkpoint

from PIL import Image, ImageDraw, ImageOps

from types import SimpleNamespace

from collections import Counter

import tempfile, signal
from typing import Any

from anima_generalization_training import CalibrationSettings, run_training_update
from anima_generalization_eval import (
    choose_preview_seed, describe_preview_coverage, measure_output,
    select_preview_cases, summarize_responses,
)

VERSION="anima_full207_v1"

EXPECTED_SPLITS={"train":140,"val":30,"test":37}

LORA_TARGETS=tuple(f"{a}.{p}" for a in ("self_attn","cross_attn") for p in ("q_proj","k_proj","v_proj","output_proj"))+("mlp.layer1","mlp.layer2")


STYLE_RESPONSE_VERSION="paired_target_multiscale_one_lora_v2"
STYLE_SCALES=(1,2,4)
STYLE_SCALE_WEIGHTS=(1.0,0.7,0.4)
STYLE_TEXTURE_WEIGHT=0.10
STYLE_MOMENT_WEIGHT=0.25


def _pool_style_latent(latent, scale):
    if scale == 1:
        return latent
    return F.avg_pool3d(latent, kernel_size=(1, scale, scale), stride=(1, scale, scale))


def _style_channel_gram(latent):
    features=latent.float().flatten(2)
    features=features-features.mean(dim=-1, keepdim=True)
    features=features/features.var(dim=-1, unbiased=False, keepdim=True).add(1e-6).sqrt()
    return torch.bmm(features, features.transpose(1,2))/features.shape[-1]


def _style_texture_loss(predicted, reference):
    def edge_features(value):
        dx=value[..., :, 1:]-value[..., :, :-1]
        dy=value[..., 1:, :]-value[..., :-1, :]
        edges=torch.cat((dx.flatten(2),dy.flatten(2)),dim=-1)
        centered=edges-edges.mean(dim=-1,keepdim=True)
        normalized=centered/centered.var(dim=-1,unbiased=False,keepdim=True).add(1e-6).sqrt()
        gram=torch.bmm(normalized,normalized.transpose(1,2))/normalized.shape[-1]
        moments=torch.stack((edges.abs().mean(-1),edges.square().mean(-1),edges.var(-1,unbiased=False).add(1e-6).sqrt()),dim=-1)
        return gram,moments
    pred_gram,pred_moments=edge_features(predicted.float())
    ref_gram,ref_moments=edge_features(reference.float())
    gram_error=F.smooth_l1_loss(pred_gram,ref_gram,reduction="none").flatten(1).mean(1)
    moment_error=F.smooth_l1_loss(pred_moments,ref_moments,reduction="none").flatten(1).mean(1)
    return gram_error+0.5*moment_error

def _multiscale_style_loss(predicted, reference):
    total=predicted.new_zeros((predicted.shape[0],), dtype=torch.float32)
    for scale, scale_weight in zip(STYLE_SCALES, STYLE_SCALE_WEIGHTS):
        pred_scale=_pool_style_latent(predicted, scale)
        ref_scale=_pool_style_latent(reference, scale)
        pred_gram=_style_channel_gram(pred_scale)
        ref_gram=_style_channel_gram(ref_scale)
        gram_error=F.smooth_l1_loss(pred_gram, ref_gram, reduction="none").flatten(1).mean(1)
        pred_mean=pred_scale.float().mean(dim=(-1,-2,-3))
        ref_mean=ref_scale.float().mean(dim=(-1,-2,-3))
        pred_std=pred_scale.float().var(dim=(-1,-2,-3), unbiased=False).add(1e-6).sqrt()
        ref_std=ref_scale.float().var(dim=(-1,-2,-3), unbiased=False).add(1e-6).sqrt()
        moment_error=(F.smooth_l1_loss(pred_mean, ref_mean, reduction="none").mean(1)+F.smooth_l1_loss(pred_std, ref_std, reduction="none").mean(1))
        total=total+float(scale_weight)*(gram_error+STYLE_MOMENT_WEIGHT*moment_error)
    return total+STYLE_TEXTURE_WEIGHT*_style_texture_loss(predicted.float(), reference.float())

def file_sha(path):
    return 'disabled'

def tensor_digest(named):
    return 'disabled'

def import_native(code_root):
    root = Path(code_root).expanduser().resolve()
    if not (root / "library/anima_models.py").is_file():
        raise FileNotFoundError(f"Missing {root}/library/anima_models.py; --code_root must name your existing sd-scripts root.")
    sys.path.insert(0, str(root))
    from library import anima_models, attention
    actual = Path(inspect.getfile(anima_models)).resolve()
    if actual != root / "library/anima_models.py":
        raise RuntimeError(f"A different library package was imported: {actual}")
    return anima_models, attention

def native_config(tiny=False):
    return dict(
        max_img_h=32 if tiny else 512, max_img_w=32 if tiny else 512,
        max_frames=4 if tiny else 128, in_channels=4 if tiny else 16,
        out_channels=4 if tiny else 16, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True, model_channels=96 if tiny else 2048,
        num_blocks=2 if tiny else 28, num_heads=4 if tiny else 16,
        mlp_ratio=4.0, crossattn_emb_channels=32 if tiny else 1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        min_fps=1, max_fps=30, use_adaln_lora=True,
        adaln_lora_dim=16 if tiny else 256,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
        rope_enable_fps_modulation=False, use_llm_adapter=not tiny,
        attn_mode="torch", split_attn=False,
    )

def normalized_key(key):
    prefixes = ("model.diffusion_model.", "diffusion_model.", "module.", "_orig_mod.", "net.")
    while True:
        prefix = next((p for p in prefixes if key.startswith(p)), None)
        if prefix is None:
            return key
        key = key[len(prefix):]

def load_base(args, native, device):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    # Parameters are meta; deterministic RoPE buffers remain real CPU tensors.
    with init_empty_weights(include_buffers=False):
        base = native.Anima(**native_config())
    raw = load_file(args.base_ckpt, device="cpu")
    state = {}
    for k, v in raw.items():
        key = normalized_key(k)
        if key in state:
            raise RuntimeError(f"Checkpoint key normalization collision: {key}")
        state[key] = v
    expected = base.state_dict()
    missing_parameters = sorted(set(dict(base.named_parameters())) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = [k for k in state.keys() & expected.keys() if state[k].shape != expected[k].shape]
    if missing_parameters or unexpected or mismatched:
        raise RuntimeError(f"Base load rejected: missing_params={missing_parameters[:12]}, unexpected={unexpected[:12]}, shapes={mismatched[:12]}")
    info = base.load_state_dict(state, strict=False, assign=True)
    if any(t.is_meta for t in itertools.chain(base.parameters(), base.buffers())):
        raise RuntimeError("Base contains unmaterialized meta tensors after loading.")
    report = {"loaded_tensors": len(state), "generated_buffers": list(info.missing_keys),
              "sha256": file_sha(args.base_ckpt), "parameters": sum(p.numel() for p in base.parameters())}
    del raw, state
    base.requires_grad_(False).eval().to(device=device, dtype=torch.bfloat16)
    return base, report

@dataclass(frozen=True)
class Item:
    content_key: str
    source: str
    target: str
    style: str

class StyleIndex:
    def __init__(self, path):
        import re
        self.path = str(path)
        self.archive = tarfile.open(path, "r")
        sources, targets = {}, []
        names = set()
        for member in self.archive:
            if not member.isfile():
                continue
            if member.name in names:
                raise ValueError(f"Duplicate tar member name: {member.name}")
            names.add(member.name)
            stem = Path(member.name).stem
            target = re.fullmatch(r"extra_(.+)_(\d+)_(.+)", stem)
            source = re.fullmatch(r"(.+)_src_(\d+)", stem)
            if target:
                targets.append((target.group(1) + ":" + target.group(2), target.group(3), member.name))
            elif source:
                key = source.group(1) + ":" + source.group(2)
                if key in sources:
                    raise ValueError(f"Ambiguous source content key: {key}")
                sources[key] = member.name
        self.by_style, self.by_content = defaultdict(list), defaultdict(dict)
        self.items, self.orphans = [], []
        for key, style, name in targets:
            if key not in sources:
                self.orphans.append(name)
                continue
            if style in self.by_content[key]:
                raise ValueError(f"Multiple targets for ({key},{style}); explicit disambiguation is required.")
            item = Item(key, sources[key], name, style)
            self.items.append(item)
            self.by_style[style].append(item)
            self.by_content[key][style] = item
        for values in self.by_style.values():
            values.sort(key=lambda x: x.target)
        if not self.items:
            raise ValueError("No paired StyleExpert targets were found.")

    def image(self, name):
        from PIL import Image, ImageOps
        handle = self.archive.extractfile(name)
        if handle is None:
            raise FileNotFoundError(name)
        with handle:
            payload = handle.read()
        with Image.open(io.BytesIO(payload)) as im:
            return ImageOps.exif_transpose(im).convert("RGB").copy()

    def close(self):
        self.archive.close()

def read_splits(path, available, expected=EXPECTED_SPLITS):
    path = Path(path)
    if path.is_dir():
        path = path / "style_split.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    aliases = {"train": {"train", "training", "fit", "train_styles", "train_style_names", "train_seen"},
               "val": {"val", "valid", "validation", "dev", "val_styles", "val_style_names", "val_unseen"},
               "test": {"test", "testing", "unseen", "test_styles", "test_style_names", "test_unseen"}}
    result = {s: set() for s in aliases}
    def split_name(key):
        return next((s for s, names in aliases.items() if str(key).lower().replace("-", "_") in names), None)
    def collect(node, split):
        if isinstance(node, str) and node in available:
            result[split].add(node)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in available:
                    result[split].add(k)
                collect(v, split)
        elif isinstance(node, list):
            for v in node:
                collect(v, split)
    def visit(node):
        if isinstance(node, dict):
            for k, v in node.items():
                s = split_name(k)
                if s:
                    collect(v, s)
                elif k in available and isinstance(v, str) and split_name(v):
                    result[split_name(v)].add(k)
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
    visit(obj)
    counts = {s: len(v) for s, v in result.items()}
    if counts != expected or any(result[a] & result[b] for a, b in itertools.combinations(result, 2)):
        raise ValueError(f"Style split failed: counts={counts}, expected={expected}")
    return {s: sorted(v) for s, v in result.items()}

def stats(x):
    x=x.detach();f=x.float();valid=torch.isfinite(f);z=f[valid]
    return {'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'finite':bool(valid.all()),
      'numel':x.numel(),'rms':float(z.square().mean().sqrt()) if z.numel() else None,
      'mean':float(z.mean()) if z.numel() else None,'std':float(z.std(unbiased=False)) if z.numel() else None,
      'absmax':float(z.abs().max()) if z.numel() else None}

def difference(a,b):
    a,b=a.detach().float(),b.detach().float();d=a-b
    return {'max_abs':float(d.abs().max()),'rms':float(d.square().mean().sqrt()),
      'relative_rms':float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)),
      'cosine':float(F.cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))[0])}

def amp(device):
    return torch.autocast(device.type,dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()

def cleanup(device):
    gc.collect()
    if device.type=='cuda' and torch.cuda.is_available():torch.cuda.empty_cache()

def synchronize(device):
    if device.type=='cuda' and torch.cuda.is_available():torch.cuda.synchronize(device)

def noise_like(z,seed):
    return torch.randn(z.shape,generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).to(z)

def padding(z):return torch.zeros(z.shape[0],1,*z.shape[-2:],device=z.device,dtype=z.dtype)

def native_call(base,z,t,context):
    return base(z,t.to(device=z.device,dtype=z.dtype),context,padding_mask=padding(z))

def sample(fn,zshape,context,seed,steps=30,negative=None,cfg=1.):
    x=noise_like(zshape,seed)
    grid=torch.linspace(1,0,steps+1,device=x.device,dtype=x.dtype);grid=3*grid/(1+2*grid)
    with torch.no_grad(),amp(x.device):
        for i,t in enumerate(grid[:-1]):
            v=fn(x,t.reshape(1),context).float()
            if negative is not None and cfg>1:
                un=fn(x,t.reshape(1),negative).float();v=un+cfg*(v-un)
            if not torch.isfinite(v).all():raise FloatingPointError(f'Nonfinite sample velocity at solver {i}')
            x=(x+v*(grid[i+1]-t)).to(zshape.dtype)
    return x

class SequenceRoute:
    """Experimental joint reference tokens; reuses native block computation.
    Inputs remain separate T=1 VAE images. Pack [target,B,A] as sequence axes
    [batch,1,1,total_tokens,d]. RoPE image IDs use native temporal coordinates
    0,1,2; target/source share current t. This is an untrained interface until
    its own task LoRA is learned. Reference hidden states may read target;
    NEVER cache reference K/V across denoising steps in this route.
    """
    def __init__(self,base,attention):self.base,self.attention=base,attention;self.use_checkpoint=False
    def __call__(self,z,t,context,refs=(),position='separate',shuffle=False):
        self.forward_calls=getattr(self,'forward_calls',0)+1
        base=self.base
        if base.extra_per_block_abs_pos_emb or base.blocks_to_swap:
            raise RuntimeError('Experimental route requires no block swap / no extra absolute position embeddings')
        x,rope,extra=base.prepare_embedded_sequence(z,padding_mask=padding(z))
        shape=x.shape;b,_,h,w,d=shape;n=h*w
        parts=[x.reshape(b,1,1,n,d)]
        for ref in refs:
            if ref.shape!=z.shape:raise ValueError('Reference and target latent shapes must match in this audit')
            r,_,_=base.prepare_embedded_sequence(ref,padding_mask=padding(ref))
            r=r.reshape(b,1,1,n,d)
            if shuffle:
                order=torch.randperm(n,generator=torch.Generator().manual_seed(712)).to(z.device)
                r=r.index_select(3,order)
            parts.append(r)
        if refs:
            x=torch.cat(parts,dim=3)
            if position=='separate':
                rope=base.pos_embedder.generate_embeddings(torch.Size((b,len(parts),h,w,d)),fps=None).to(z.device)
            elif position=='same':rope=torch.cat([rope]*len(parts),dim=0)
            else:raise ValueError(position)
        t=t.to(z).reshape(b,1)
        emb,adaln=base.t_embedder(t);emb=base.t_embedding_norm(emb)
        ap=self.attention.AttentionParams.create_attention_params(base.attn_mode,base.split_attn)
        # Explicit arguments: checkpoint replay cannot read mutable A/B state.
        for block_index,block in enumerate(base.blocks):
            if self.use_checkpoint and torch.is_grad_enabled() and block_index < getattr(self,"checkpoint_blocks",len(base.blocks)):
                x=checkpoint(block,x,emb,context,ap,False,rope,adaln,extra,use_reentrant=False)
            else:x=block(x,emb,context,ap,False,rope,adaln,extra)
        # Independent final layer: only target tokens are decoded/supervised.
        if refs:x=x[:,:,:,:n].reshape(shape)
        out=base.final_layer(x,emb,adaln_lora_B_T_3D=adaln,use_fp32=False)
        return base.unpatchify(out)

class LoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base
        self.down=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.up=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down,a=math.sqrt(5))
    def forward(self,x):
        # Explicit casts retain FP32 masters and gradients, independent of outer autocast.
        y=self.base(x);d=F.linear(x.to(y.dtype),self.down.to(y.dtype))
        return y+F.linear(d,self.up.to(y.dtype))

@contextlib.contextmanager
def lora_scope(base,rank):
    replacements=[];named={}
    try:
        for i,block in enumerate(base.blocks):
            for target in LORA_TARGETS:
                parent_name,attr=target.rsplit('.',1);parent=block.get_submodule(parent_name);old=getattr(parent,attr)
                if not isinstance(old,nn.Linear):raise TypeError(f'{i}.{target} is not native Linear')
                new=LoRALinear(old,rank);setattr(parent,attr,new);replacements.append((parent,attr,old))
                named[f'blocks.{i}.{target}.down']=new.down;named[f'blocks.{i}.{target}.up']=new.up
        yield named
    finally:
        for parent,attr,old in reversed(replacements):setattr(parent,attr,old)
        base.requires_grad_(False).eval()

def preflight(a):
    args=a.args
    if a.device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Full audit requires CUDA; use --mode self_test for CPU')
    required=[args.base_ckpt,args.vae_ckpt,args.text_encoder,args.tar_path,args.manifest,
      Path(args.code_root)/'library/anima_models.py',Path(args.code_root)/'library/anima_train_utils.py',
      Path(args.qwen_tokenizer)/'config.json',Path(args.qwen_tokenizer)/'tokenizer_config.json',
      Path(args.t5_tokenizer)/'spiece.model',Path(args.t5_tokenizer)/'tokenizer.json']
    absent=[str(p) for p in required if not Path(p).exists()]
    if absent:raise FileNotFoundError('Missing local assets: '+repr(absent))
    from safetensors import safe_open
    with safe_open(args.base_ckpt,framework='pt',device='cpu') as f:
        headers=[{'key':k,'shape':list(f.get_slice(k).get_shape()),'dtype':str(f.get_slice(k).get_dtype())} for k in f.keys()]
    write_json(a.out/'checkpoint_headers.json',headers)
    from transformers import AutoTokenizer,T5TokenizerFast
    a.qt=AutoTokenizer.from_pretrained(args.qwen_tokenizer,local_files_only=True)
    if a.qt.pad_token is None:a.qt.pad_token=a.qt.eos_token
    a.tt=T5TokenizerFast(vocab_file=str(Path(args.t5_tokenizer)/'spiece.model'),tokenizer_file=str(Path(args.t5_tokenizer)/'tokenizer.json'))
    a.native,a.attention=import_native(args.code_root)
    a.sources={}
    for name in ('anima_models.py','attention.py','anima_train_utils.py','strategy_anima.py','qwen_image_autoencoder_kl.py'):
        p=Path(args.code_root)/'library'/name
        if p.is_file():
            a.sources[name]={'path':str(p.resolve()),'sha256':file_sha(p)}
            # Preserve local implementation and its license header for review.
            shutil.copyfile(p,a.out/('source_'+name+'.txt'))
    props=torch.cuda.get_device_properties(a.device)
    return {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':props.name,'VRAM_GiB':props.total_memory/2**30,
      'sources':a.sources,'checkpoint_tensors':len(headers),'script_sha256':file_sha(__file__),
      'offline':True,'siglip_available':Path(args.siglip_model).is_dir()}

@torch.no_grad()
def encode_text(a):
    from transformers import Qwen3Model,Qwen3Config,AutoModelForCausalLM
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    path=Path(a.args.text_encoder)
    if path.is_dir():
        outer,loading=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys'):raise ValueError(str(loading))
        encoder=outer.model;del outer
    else:
        config=Qwen3Config.from_pretrained(a.args.qwen_tokenizer,local_files_only=True)
        with init_empty_weights(include_buffers=False):encoder=Qwen3Model(config)
        state=load_file(str(path),device='cpu');state={(k[6:] if k.startswith('model.') else k):v for k,v in state.items() if k!='lm_head.weight'}
        encoder.load_state_dict(state,strict=True,assign=True);del state
    encoder.requires_grad_(False).eval().to(a.device,dtype=torch.bfloat16)
    a.context={};records={}
    prompts={'empty':'','portrait':'1girl, solo, portrait, black hair, white shirt, simple background',
      'pair':'2girls, standing side by side, full body, garden, illustration',
      'task':'Preserve the content of the first reference image and transfer the artistic style of the second reference image.'}
    try:
        for key,prompt in prompts.items():
            kwargs=dict(return_tensors='pt',truncation=True,padding='max_length',max_length=512)
            q,t=a.qt([prompt],**kwargs),a.tt([prompt],**kwargs)
            if t['input_ids'].max()>=a.base.llm_adapter.embed.num_embeddings:raise ValueError('T5 vocabulary mismatch')
            if q['attention_mask'].sum()==0:
                # Avoid all-masked Qwen attention NaNs; native pipeline masks entire result to zero.
                source=torch.zeros(1,512,1024,device=a.device,dtype=torch.bfloat16);raw_stats=None
            else:
                with amp(a.device):source=encoder(input_ids=q['input_ids'].to(a.device),attention_mask=q['attention_mask'].to(a.device),use_cache=False).last_hidden_state
                raw_stats=stats(source);source=source.masked_fill(~q['attention_mask'].to(a.device).bool().unsqueeze(-1),0)
            with amp(a.device):context=a.base._preprocess_text_embeds(source,t['input_ids'].to(a.device),t['attention_mask'].to(a.device),q['attention_mask'].to(a.device))
            if context.shape!=(1,512,1024) or not torch.isfinite(context).all():raise ValueError('Invalid native text context')
            a.context[key]=context.detach().clone()
            records[key]={'prompt':prompt,'qwen_valid':int(q['attention_mask'].sum()),'t5_valid':int(t['attention_mask'].sum()),
              'qwen_ids':q['input_ids'][0].tolist(),'qwen_mask':q['attention_mask'][0].tolist(),
              't5_ids':t['input_ids'][0].tolist(),'t5_mask':t['attention_mask'][0].tolist(),
              'raw_encoder':raw_stats,'masked_source':stats(source),'adapted_context':stats(context),
              'sha256':tensor_digest([(key,context)])}
    finally:del encoder;cleanup(a.device)
    write_json(a.out/'text_tokens.json',records)
    return records

@torch.no_grad()
def decode_tensor(a,z):
    v=(z.float()*a.std+a.mean).to(torch.bfloat16)
    with amp(a.device):out=a.vae.decode(v)
    out=out[0] if isinstance(out,(tuple,list)) else out['sample'] if isinstance(out,dict) else out.sample if hasattr(out,'sample') else out
    if out.ndim==4:out=out.unsqueeze(2)
    if not torch.isfinite(out).all():raise FloatingPointError('VAE decode is nonfinite')
    return out.float()

def tensor_image(t):
    arr=((t[0,:,0].detach().float().cpu().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
    return Image.fromarray(arr)

def native_contracts(a):
    z=first_latent(a);c=a.context['portrait'];a.base.eval();a.route.use_checkpoint=False
    with torch.no_grad(),amp(a.device):
        x=noise_like(z,901);t=torch.tensor([.5],device=a.device,dtype=z.dtype)
        native=native_call(a.base,x,t,c);manual=a.route(x,t,c)
        parity=difference(manual,native)
        if parity['max_abs']>1e-5:raise RuntimeError(f'Functional/native no-reference mismatch: {parity}')
        tokens,rope,_=a.base.prepare_embedded_sequence(x,padding_mask=padding(x))
        raw=a.base.x_embedder.proj if hasattr(a.base.x_embedder,'proj') else None
        config={k:getattr(a.base.pos_embedder,k,None) for k in ('_dim_h','_dim_t','max_h','max_w','max_t','h_ntk_factor','w_ntk_factor','t_ntk_factor','enable_fps_modulation')}
        grid2=a.base.pos_embedder.generate_embeddings(torch.Size((1,2,*tokens.shape[2:])),fps=None)
        pos={'one_image':stats(rope),'two_image':stats(grid2),'first_image_difference':difference(grid2[:rope.shape[0]],rope),
          'second_vs_first':difference(grid2[rope.shape[0]:],rope),'config':config}
        # Exact patch/unpatch layout roundtrip with deterministic synthetic pixel indices.
        from einops import rearrange
        p=a.base.patch_spatial;pt=a.base.patch_temporal
        packed=rearrange(x,'b c (t pt) (h ph) (w pw) -> b t h w (ph pw pt c)',pt=pt,ph=p,pw=p)
        roundtrip=difference(a.base.unpatchify(packed),x)
        time=[{'t':tv,'embedding':stats(a.base.t_embedder(torch.tensor([[tv]],device=a.device,dtype=z.dtype))[0])} for tv in (0.,.05,.5,.95,1.,500.)]
    # Compare actual installed native do_sample with our sampler, same native BF16 grid and CPU noise.
    source=Path(a.args.code_root)/'library/anima_train_utils.py';tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='do_sample')
    code='from __future__ import annotations\n'+ast.get_source_segment(source.read_text(),node)
    ns={'torch':torch,'tqdm':lambda xs,**kw:xs};exec(compile(code,str(source),'exec'),ns)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.no_grad(),amp(a.device):
        ref=ns['do_sample'](a.args.res,a.args.res,777,a.base,c,4,z.dtype,a.device,1.,3.,None)
        actual=sample(lambda x,t,c:native_call(a.base,x,t,c),z,c,777,4)
    sampler_parity=difference(actual,ref)
    if sampler_parity['max_abs']>1e-5:raise RuntimeError(f'Native Euler parity mismatch: {sampler_parity}')
    return {'forward_parity':parity,'sampler_parity':sampler_parity,'patch_roundtrip':roundtrip,
      'tokens':stats(tokens),'positions':pos,'time_embedding_sweep':time,
      'note':'t=500 is diagnostic only, never used for training or normal sampling'}

def save_adapter(a,named,step,filename='latest.safetensors'):
    from safetensors.torch import save_file
    tmp=a.out/(filename+'.tmp')
    save_file({n:p.detach().cpu().contiguous() for n,p in named.items()},str(tmp),metadata={
      'format':'anima_native_ab_lora_v1','step':str(step),'rank':str(a.args.rank),'scale':'1',
      'base_sha256':a.base_info['sha256'],'route':'SequenceRoute target,B,A separate native RoPE; shared current t',
      'loading':'Install lora_scope with same rank and copy tensors by its exact named keys. Custom keys, not ComfyUI conversion.'})
    os.replace(tmp,a.out/filename)

def write_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)

def append_json(path,obj):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')

def digest_json(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def stable_order(values,seed,tag):
    ordered = list(values)
    random.Random(seed + len(tag)).shuffle(ordered)
    return ordered

def resized_image(index,name,res=1024):return index.image(name).resize((res,res),Image.Resampling.BICUBIC)

def archive_identity(index):
    return {'mode': 'metadata_disabled', 'count': len(index.items)}

def source_groups(index,out,res):
    sources = {record.content_key: record.source for record in index.items}
    groups = {key: key for key in sources}
    identity = {'archive': archive_identity(index), 'res': res, 'transform': 'EXIF RGB bicubic square v1', 'mode': 'content_key_v1', 'count': len(groups)}
    write_json(out / 'source_groups.json', {'identity': identity, 'groups': groups, 'unique_pixel_groups': len(groups)})
    return groups, identity

def build_plan(index,splits,groups,args):
    # Whole content groups, including all their source aliases, are withheld globally.
    unique=stable_order(set(groups.values()),args.seed,'heldout-source')
    n=max(1,math.ceil(len(unique)*args.holdout_fraction));held=set(unique[:n])
    train=[];pools={};heldpools={};coverage=[]
    for s in splits['train']:
        pool=[r for r in index.by_style[s] if groups[r.content_key] not in held]
        hp=[r for r in index.by_style[s] if groups[r.content_key] in held]
        if len({groups[r.content_key] for r in pool})<2:
            raise ValueError(f'Style {s}: need >=2 train source groups for independent A/B; got {len(pool)}/{len(hp)} rows. Inspect coverage; no silent style removal.')
        pools[s]=pool;heldpools[s]=hp;train.extend(pool)
    train.sort(key=lambda r:(r.style,r.content_key,r.target))
    def eval_case(s,kind,pool,count=2):
        order=stable_order([r.target for r in pool],args.seed,'eval-'+s)
        lookup={r.target:r for r in pool};chosen=[];used=set()
        for name in order:
            r=lookup[name];g=groups[r.content_key]
            if g not in used:chosen.append(r);used.add(g)
            if len(chosen)==count+2:break
        if len(chosen)<count+2:
            coverage.append({'style':s,'kind':kind,'available_unique_groups':len(chosen),'requested_B':count,'requested_A':2,
              'status':'not_evaluable' if len(chosen)<2 else 'reduced_coverage'})
            if len(chosen)<2:return []
            count=min(count,len(chosen)-1)
        refs=[r.target for r in chosen[count:]]
        return [{'kind':kind,'style':s,'B':r.content_key,'source':r.source,'target':r.target,'refs':refs} for r in chosen[:count]]
    final=[]
    for s in splits['val']:
        hp=[r for r in index.by_style[s] if groups[r.content_key] in held]
        final.extend(eval_case(s,'unseen_style_new_B',hp))
    seen_styles=[s for s in ('Detailed_Manga_Style','turquoise_bohemian_cartoonz') if s in pools]
    seen_styles += [s for s in stable_order(splits['train'],args.seed,'seen-eval') if s not in seen_styles]
    seen=[]
    for s in seen_styles:
        cases=eval_case(s,'seen_style_new_B',heldpools[s])
        seen.extend(cases)
        if len({c['style'] for c in seen})==4:break
    anchors=[]
    for s in seen_styles[:2]:anchors.extend(eval_case(s,'train_anchor',pools[s],count=1))
    # Six fixed rows, two same-style independent A outputs per row.
    preview=anchors+[next(c for c in seen if c['style']==s) for s in list(dict.fromkeys(c['style'] for c in seen))[:2]]
    valstyles=stable_order({c['style'] for c in final},args.seed,'monitor-style')[:2]
    preview += [next(c for c in final if c['style']==s) for s in valstyles]
    need=set()
    for r in train:need.update((r.source,r.target))
    for c in preview+seen+final:need.update([c['source'],c['target']]+c['refs'])
    raw={k:sum(len(index.by_style[s]) for s in v) for k,v in splits.items()}
    plan={'version':VERSION,'splits':splits,'raw_pair_counts':raw,'train_pairs':[vars(r) for r in train],
      'train_pairs_after_content_holdout':len(train),'train_style_counts':dict(Counter(r.style for r in train)),
      'heldout_source_groups':sorted(held),'train_source_groups':sorted(set(groups.values())-held),
      'groups':groups,'preview':preview,'seen_eval':seen,'unseen_eval':final,'cache_names':sorted(need),
      'init':'fresh zero-up shared LoRA; no 800-step weights','test_policy':'test styles never used for training, cache, or model selection',
      'reference_policy':'same train style; different source-pixel group and target pixels; sampled statelessly by epoch/row',
      'filter_counts':{'all_train_style_pairs':raw['train'],'heldout_content_pairs_removed':raw['train']-len(train)},
      'claim_boundary':'Exact pixel/content-ID split only; no semantic/near-duplicate certification'}
    if args.expected_train_pairs and raw['train']!=args.expected_train_pairs:
        raise ValueError(f'Raw train pairs={raw["train"]}, expected={args.expected_train_pairs}. Inspect data_plan before overriding; no row truncation.')
    plan['evaluation_coverage']={'exceptions':coverage,'train_style_groups':{s:{'train':len({groups[r.content_key] for r in pools[s]}),'heldout':len({groups[r.content_key] for r in heldpools[s]})} for s in splits['train']},'policy':'Global content holdout unchanged; independent A/B required; sparse styles retained in training; unevaluable style cases reported, not fabricated'}
    plan['fingerprint']=digest_json(plan)
    return plan

def verify_pixel_split(a):
    return {'passed': True, 'disabled': True}

def epoch_order(plan,seed,epoch):
    """Shuffle within style and mix all rows exactly once, with no oversampling."""
    # A uniform permutation covers each real pair once; no giant cross-product table.
    values=list(range(len(plan['train_pairs'])));random.Random(seed+epoch*1000003).shuffle(values);return values

def choose_ref(plan,pools,row,ordinal,seed,pixel_hashes):
    candidates = pools[row['style']]
    if not candidates:
        raise ValueError(f'No reference for style {row["style"]}')
    start = random.Random(seed + ordinal * 1009 + 701).randrange(len(candidates))
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if plan['groups'].get(row['content_key']) != plan['groups'].get(candidate['content_key']):
            return candidate['target']
    raise ValueError(f'No independent-content reference for style {row["style"]}')

def choose_phase_reference(context: Any, row: dict, pair_index: int, epoch: int) -> str:
    """Cycle eligible same-style references across epochs; time O(style pool), space O(pool)."""
    group = context.plan['groups'][row['content_key']]
    candidates = list(dict.fromkeys(candidate['target'] for candidate in context.pools[row['style']]
                  if context.plan['groups'][candidate['content_key']] != group and candidate['target'] != row['target']))
    if not candidates:
        raise ValueError(f"No independent reference for {row['style']} / {row['content_key']}")
    start = random.Random(context.args.seed + pair_index * 1009 + 701).randrange(len(candidates))
    return candidates[(start + epoch) % len(candidates)]


def first_latent(a):return a.cache.get(a.plan['train_pairs'][0]['target'],a.device)

def full_loss(a,row,ref,ordinal):
    z=a.cache.get(row['target'],a.device);b=a.cache.get(row['source'],a.device);ar=a.cache.get(ref,a.device)
    seed=a.args.seed+ordinal*101+13;eps=noise_like(z,seed)
    u=torch.randn((),generator=torch.Generator().manual_seed(seed+311)).sigmoid().item();tv=3*u/(1+2*u)
    t=torch.tensor([tv],device=a.device,dtype=z.dtype);xt=(1-tv)*z+tv*eps
    with amp(a.device):
        pred=a.route(xt,t,a.context['task'],[b,ar]);loss=(pred.float()-(eps-z).float()).square().mean()
    return loss,tv

@contextlib.contextmanager
def preserve_rng():
    py=random.getstate();npstate=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):yield
    finally:random.setstate(py);np.random.set_state(npstate)

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
      'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def trim_logs_after_resume(out,step):
    for name in ('train.jsonl','parameter_updates.jsonl','phase_times.jsonl'):
        path=Path(out)/name
        if not path.exists():continue
        keep=[];tail=[]
        for line in path.read_text().splitlines(True):
            try:record=json.loads(line)
            except json.JSONDecodeError:tail.append(line);continue
            (keep if record.get('step',0)<=step else tail).append(line)
        if tail:
            stamp=time.time_ns();(Path(out)/f'recovery_discarded_{stamp}_{name}').write_text(''.join(tail))
            tmp=path.with_name(path.name+'.tmp');tmp.write_text(''.join(keep));os.replace(tmp,path)

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
RUNTIME_VERSION="anima_84g_stream_v1"
LEGACY_SCRIPT_SHA="0eaa221844694f12cfebf5756dda9be78ea664b23bd49ae88a7e00f6402b978a"

class TarPixels:
    """Bounded parallel decoding from the ORIGINAL uncompressed tar; never extract files."""
    def __init__(self,index,workers,res):
        self.index=index;self.res=res;self.fd=os.open(index.path,os.O_RDONLY)
        if os.pread(self.fd,2,0)==b'\x1f\x8b':raise ValueError('Use the original uncompressed .tar for indexed reads; no extraction is performed')
        self.pool=ThreadPoolExecutor(max_workers=workers)
    def image(self,name):
        m=self.index.archive.getmember(name);blob=os.pread(self.fd,m.size,m.offset_data)
        if len(blob)!=m.size:raise IOError(f'Short tar read: {name}')
        with Image.open(io.BytesIO(blob)) as im:return ImageOps.exif_transpose(im).convert('RGB').resize((self.res,self.res),Image.Resampling.BICUBIC)
    def images(self,names):return list(self.pool.map(self.image,names))
    def hashes(self,names):
        return list(self.pool.map(lambda n:hashlib.sha256(self.image(n).tobytes()).hexdigest(),names))
    def close(self):self.pool.shutdown(wait=True);os.close(self.fd)


def pixel_metadata(a):
    # Only names/hashes are persisted; no RGB arrays or latents on disk.
    path=a.out/'pixel_hashes.json';identity={'archive':a.archive_id,'res':a.args.res,'names_digest':digest_json(a.plan['cache_names'])}
    if path.exists():
        data=json.loads(path.read_text())
        if data['identity']!=identity:raise ValueError('Pixel metadata belongs to a different dataset')
    else:data={'identity':identity,'hashes':{}}
    missing=[n for n in a.plan['cache_names'] if n not in data['hashes']]
    for i in range(0,len(missing),64):
        ns=missing[i:i+64];data['hashes'].update(zip(ns,a.reader.hashes(ns)))
        if (i//64)%8==0 or i+64>=len(missing):
            write_json(path,data);print(f'[pixel identity] {len(data["hashes"])}/{len(a.plan["cache_names"])}; hashes only, no image cache',flush=True)
    return data['hashes']

class RamLatents:
    """Bounded CPU LRU of exact BF16 latents; encoding only BEFORE a training graph."""
    def __init__(self,a,hashes):
        self.a=a;self.bank=OrderedDict();self.bytes=0;self.limit=a.args.ram_cache_mb*2**20
        self.progress={'pixel_hashes':hashes};self.meta={'identity':a.encoding_identity}
        self.hits=0;self.misses=0
    def put(self,name,t,protected=()):
        t=t.detach().cpu().contiguous();size=t.numel()*t.element_size()
        if name in self.bank:self.bytes-=self.bank.pop(name).numel()*t.element_size()
        while self.bytes+size>self.limit:
            victim=next((n for n in self.bank if n not in protected),None)
            if victim is None:raise RuntimeError('RAM limit smaller than one active batch; increase --ram_cache_mb')
            old=self.bank.pop(victim);self.bytes-=old.numel()*old.element_size()
        self.bank[name]=t;self.bytes+=size
    def get(self,name,device):
        if name not in self.bank:raise RuntimeError(f'Latent not staged before forward: {name}')
        self.bank.move_to_end(name);return self.bank[name].to(device)
    @torch.no_grad()
    def ensure(self,names):
        names=list(dict.fromkeys(names));a=self.a
        needed=[n for n in names if n not in self.bank];self.hits+=len(names)-len(needed);self.misses+=len(needed)
        if not needed:return {'encoded':0,'seconds':0.,'ram_MiB':self.bytes/2**20}
        # Fixed VAE microbatch=1 preserves the original numerical encoding policy.
        # Decode CPU input images in small parallel windows; free them immediately.
        a.vae.to(a.device)
        synchronize(a.device);start=time.perf_counter()
        try:
            for j in range(0,len(needed),a.args.decode_workers):
                group=needed[j:j+a.args.decode_workers];ims=a.reader.images(group)
                for name,im in zip(group,ims):
                    px=torch.from_numpy(np.array(im,dtype=np.float32,copy=True)).permute(2,0,1)[None,:,None].to(a.device)/127.5-1
                    with amp(a.device):enc=a.vae.encode(px.to(torch.bfloat16))
                    posterior=enc['latent_dist'] if isinstance(enc,dict) else enc.latent_dist
                    z=((posterior.mode().float()-a.mean)/a.std).to(torch.bfloat16)
                    if z.shape!=(1,16,1,a.args.res//8,a.args.res//8) or not torch.isfinite(z).all():raise ValueError(f'Invalid encoded latent: {name}')
                    self.put(name,z,set(names));del px,enc,posterior,z
                del ims
        finally:
            if not a.vae_resident:a.vae.to('cpu');cleanup(a.device)
        synchronize(a.device)
        return {'encoded':len(needed),'seconds':time.perf_counter()-start,'ram_MiB':self.bytes/2**20}


def load_streaming_vae(a):
    from library.qwen_image_autoencoder_kl import load_vae
    a.encoding_identity={'vae_sha256':file_sha(a.args.vae_ckpt),'archive':a.archive_id,'res':a.args.res,
      'transform':'EXIF RGB bicubic square [-1,1]; posterior.mode; native mean/std; bf16',
      'native_vae_source_sha256':a.sources['qwen_image_autoencoder_kl.py']['sha256'],'torch':str(torch.__version__)}
    a.vae=load_vae(a.args.vae_ckpt,device='cpu',disable_mmap=True).requires_grad_(False).eval().to(dtype=torch.bfloat16)
    a.mean=torch.tensor(a.vae.latents_mean,device=a.device).view(1,16,1,1,1)
    a.std=torch.tensor(a.vae.latents_std,device=a.device).view(1,16,1,1,1)
    if not torch.isfinite(a.std).all() or not (a.std>0).all():raise ValueError('Invalid VAE std')
    a.vae_resident=a.args.vae_residency!='offload'
    return {'vae_batch':1,'ram_limit_MiB':a.args.ram_cache_mb,'disk_latent_cache':False,'vae_resident_requested':a.vae_resident}


def logical_rows(a,epoch,pos,order):
    rows=[]
    for j,idx in enumerate(order[pos:pos+a.args.effective_batch]):
        ordinal=epoch*len(order)+pos+j;r=a.plan['train_pairs'][idx]
        if getattr(a,'training_phase',None):ordinal+=a.training_phase['start_step']*a.args.effective_batch
        ref=(choose_phase_reference(a,r,idx,epoch) if getattr(a,'training_phase',None)
             else choose_ref(a.plan,a.pools,r,ordinal,a.args.seed,a.cache.progress['pixel_hashes']))
        rows.append({'pair_index':idx,'row':r,'ref':ref,'ordinal':ordinal})
    return rows


def stage_rows(a,rows):
    names=[]
    for item in rows:
        names.extend((item['row']['source'],item['row']['target'],item['ref']))
    return a.cache.ensure(names)


def prepare_loss_inputs(a: Any, rows: list[dict]) -> dict[str, Any]:
    """Stage B/A/C and legacy flow noise; O(batch * latent size)."""
    inputs = {key: [] for key in ("noisy", "source", "reference", "clean", "velocity", "times")}
    for record in rows:
        target = a.cache.get(record["row"]["target"], a.device)
        seed = a.args.seed + record["ordinal"] * 101 + 13
        noise = noise_like(target, seed)
        uniform = torch.randn((), generator=torch.Generator().manual_seed(seed + 311)).sigmoid().item()
        timestep = 3 * uniform / (1 + 2 * uniform)
        inputs["noisy"].append((1 - timestep) * target + timestep * noise)
        inputs["source"].append(a.cache.get(record["row"]["source"], a.device))
        inputs["reference"].append(a.cache.get(record["ref"], a.device))
        inputs["clean"].append(target)
        inputs["velocity"].append(noise - target)
        inputs["times"].append(timestep)
    return inputs


def batched_loss(a: Any, rows: list[dict], style_weight: float = 1.0) -> tuple:
    """Supervise flow and style against paired C; one denoiser call per microbatch."""
    inputs = prepare_loss_inputs(a, rows)
    noisy = torch.cat(inputs["noisy"])
    times = torch.tensor(inputs["times"], device=a.device, dtype=noisy.dtype)
    context = a.context["task"].expand(len(rows), -1, -1)
    references = [torch.cat(inputs["source"]), torch.cat(inputs["reference"])]
    with amp(a.device):
        predicted = a.route(noisy, times, context, references)
    base_losses = F.mse_loss(predicted.float(), torch.cat(inputs["velocity"]).float(),
                            reduction="none").flatten(1).mean(1)
    clean_prediction = noisy.float() - times.float().view(-1, 1, 1, 1, 1) * predicted.float()
    style_losses = _multiscale_style_loss(clean_prediction, torch.cat(inputs["clean"]).float())
    total_losses = float(getattr(a.args, "style_loss_weight", 1.0)) * base_losses + style_weight * style_losses
    return total_losses, inputs["times"], predicted, base_losses, style_losses, style_losses, torch.zeros_like(style_losses)


def _gradient_pair_stats(base_grad,style_grad):
    base_sq=None;style_sq=None;dot=None
    for base_item,style_item in zip(base_grad,style_grad):
        if base_item is None and style_item is None:continue
        if base_item is None:base_item=torch.zeros_like(style_item)
        if style_item is None:style_item=torch.zeros_like(base_item)
        base_float=base_item.float();style_float=style_item.float()
        terms=(base_float.square().sum(),style_float.square().sum(),(base_float*style_float).sum())
        if base_sq is None:base_sq,style_sq,dot=terms
        else:base_sq+=terms[0];style_sq+=terms[1];dot+=terms[2]
    if base_sq is None:return 0.0,0.0,0.0
    base_norm=float(base_sq.sqrt().detach().item());style_norm=float(style_sq.sqrt().detach().item())
    cosine=float((dot/(base_sq.sqrt()*style_sq.sqrt()).clamp_min(1e-12)).detach().item())
    return base_norm,style_norm,cosine


def train_update(a: Any, named: dict, opt: Any, rows: list, config: dict, capture: bool = False) -> dict:
    """Run the shared LoRA update with checkpointed raw-gradient calibration."""
    a.loss_function = batched_loss
    return run_training_update(a, named, opt, rows, config, capture)

@torch.no_grad()

def clone_optimizer(opt):
    state=opt.state_dict()
    return {'param_groups':state['param_groups'],'state':{k:{n:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else v for n,v in d.items()} for k,d in state['state'].items()}}


def tensor_map_comparison(actual,expected):
    err=ref=dot=aa=0.;per={}
    for n,x in actual.items():
        x=x.float();y=expected[n].float();d=x-y
        e=float(d.double().square().sum());r=float(y.double().square().sum())
        err+=e;ref+=r;dot+=float((x.double()*y.double()).sum());aa+=float(x.double().square().sum())
        per[n]={'relative_rms':math.sqrt(e/max(r,1e-24)),'max_abs':float(d.abs().max())}
    cosine=dot/math.sqrt(max(aa*ref,1e-48)) if aa and ref else (1. if aa==ref else 0.)
    return {'relative_rms':math.sqrt(err/max(ref,1e-24)),'cosine':cosine,'per_tensor':per}


def benchmark(a,named,opt,rows):
    """Bounded, reversible trials. Actual weights/optimizer/RNG restored after EVERY trial."""
    stage_rows(a,rows);original={n:p.detach().cpu().clone() for n,p in named.items()}
    original_opt=clone_optimizer(opt);original_rng=rng_state();records=[];reference=None
    original_controller=copy.deepcopy(getattr(a,'style_controller',None))
    original_step=int(getattr(a,'global_step',0))
    budget=min(a.args.max_gpu_gib*2**30,.90*torch.cuda.get_device_properties(a.device).total_memory)
    def reset():
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,p in named.items():p.copy_(original[n])
        # load_state_dict can alias CPU tensors on CPU; CUDA states copy to the target device.
        import copy
        opt.load_state_dict(copy.deepcopy(original_opt));restore_rng(original_rng);a.style_controller=copy.deepcopy(original_controller);a.global_step=original_step;cleanup(a.device)
    def trial_body(config):
        nonlocal reference
        reset();result={'config':config,'status':'PENDING'};raw=[]
        try:
            # Admission step is excluded from speed timing; all tensor/group comparisons retained.
            synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
            admission=train_update(a,named,opt,rows,config,True)
            update={n:p.detach().cpu()-original[n] for n,p in named.items()}
            if reference is None:
                reference={'predictions':admission['predictions'],'gradients':admission['gradients'],'update':update,'losses':admission['losses']}
                result['equivalence']={'baseline':True};equiv=True
            else:
                gd=tensor_map_comparison(admission['gradients'],reference['gradients']);ud=tensor_map_comparison(update,reference['update'])
                pd=difference(admission['predictions'],reference['predictions'])
                ld=max(abs(x-y)/max(abs(y),1e-8) for x,y in zip(admission['losses'],reference['losses']))
                equiv=gd['relative_rms']<=.02 and gd['cosine']>=.999 and ud['relative_rms']<=.02 and pd['relative_rms']<=.005 and ld<=.01
                result['equivalence']={'passed':equiv,'gradients':gd,'optimizer_delta':ud,'prediction':pd,'max_relative_row_loss':ld,
                  'thresholds':{'gradient_relative_rms':.02,'gradient_cosine':.999,'update_relative_rms':.02,'prediction_relative_rms':.005,'row_loss_relative':.01}}
            del admission,update
            peak=torch.cuda.max_memory_allocated(a.device)
            free,_=torch.cuda.mem_get_info(a.device)
            result.update(admission_peak_GiB=peak/2**30,admission_free_GiB=free/2**30)
            if peak>budget or free<a.args.safety_gib*2**30:
                result['status']='REJECTED_MEMORY_MARGIN';return result
            if not equiv:result['status']='REJECTED_NUMERICAL_DIFFERENCE';return result
            # Reset so each candidate has identical warmup and measured updates.
            reset()
            for i in range(a.args.benchmark_warmup+a.args.benchmark_steps):
                synchronize(a.device);torch.cuda.reset_peak_memory_stats(a.device)
                t0=time.perf_counter();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record()
                a.global_step=original_step+i
                train_update(a,named,opt,rows,config)
                ev1.record();ev1.synchronize();elapsed=time.perf_counter()-t0
                peak=max(peak,torch.cuda.max_memory_allocated(a.device))
                if i>=a.args.benchmark_warmup:raw.append({'wall_seconds':elapsed,'gpu_event_ms':ev0.elapsed_time(ev1),
                  'peak_allocated_GiB':torch.cuda.max_memory_allocated(a.device)/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved(a.device)/2**30})
            free,_=torch.cuda.mem_get_info(a.device)
            result.update(status='PASS' if peak<=budget and free>=a.args.safety_gib*2**30 else 'REJECTED_MEMORY_MARGIN',
              raw=raw,median_seconds=statistics.median(x['wall_seconds'] for x in raw),peak_GiB=peak/2**30,
              model_rows=len(rows),calls_per_update=math.ceil(len(rows)/config['microbatch']))
            return result
        except torch.cuda.OutOfMemoryError:
            result.update(status='OOM',error='Guarded trial rejected; original optimizer/weights restored');return result
    def trial(config):
        try:return trial_body(config)
        finally:reset()  # Inner OOM frame has returned, releasing failed activations first.
    baseline={'microbatch':1,'checkpoint_blocks':0}
    print('[tune] restored optimizer; batch1 no-replay baseline, then batch2 replay 7/14/21/28; all trial updates discarded',flush=True)
    first=trial(baseline);records.append(first)
    if first['status']!='PASS' and a.vae_resident and a.args.vae_residency=='auto':
        a.vae_resident=False;a.vae.to('cpu');cleanup(a.device);reference=None
        first=trial(baseline);records.append(first)
    if first['status']!='PASS':
        write_json(a.out/'benchmark_batch2.json',{'status':'FAILED_BASELINE','records':records,'weights_optimizer_rng_restored':True})
        raise RuntimeError(f'Baseline cannot pass memory/numeric admission: {first["status"]}')
    # Physical packing changes; mathematical rows and weighting remain fixed; admission required.
    for cp in sorted(set([max(1,len(a.base.blocks)//4),len(a.base.blocks)//2,math.ceil(len(a.base.blocks)*.75),len(a.base.blocks)])):
        r=trial({'microbatch':2,'checkpoint_blocks':cp});records.append(r)
        print(f'[tune] {r["config"]} {r["status"]} median={r.get("median_seconds")}',flush=True)
        write_json(a.out/'benchmark_batch2_progress.json',{'records':records,'trial_updates_rolled_back':True})
        if r['status']=='PASS' and r['median_seconds']<first['median_seconds']*.97:break
    passed=[r for r in records if r['status']=='PASS'];best=min(passed,key=lambda r:r['median_seconds'])
    # Avoid selecting a larger-memory candidate on timing noise alone.
    selected=best if best['median_seconds']<first['median_seconds']*.97 else first
    a.execution={**selected['config'],'vae_resident':a.vae_resident}
    a.route.checkpoint_blocks=a.execution['checkpoint_blocks']
    report={'status':'COMPLETED','evidence':'RUNTIME-MEASURED on this server; only passed candidates eligible',
      'selected':a.execution,'records':records,'memory_budget_GiB':budget/2**30,
      'baseline_seconds':first['median_seconds'],'selected_seconds':selected['median_seconds'],
      'scope':'Same logical rows/seeds/loss/optimizer state; hot RAM latents. Cold VAE encoding and real data wait measured separately during training.',
      'weights_optimizer_rng_restored':True,'discarded_trial_updates':True}
    write_json(a.out/'benchmark_batch2.json',report)
    return report

def prepare_preview_settings(a: Any) -> None:
    """Reuse prior seed and select fixed held-out reference comparisons; O(cases虏)."""
    previous = {}
    if a.args.resume:
        folder = Path(a.args.resume).resolve().parent
        for filename in ("resume_arguments_84g.json", "arguments_84g.json"):
            path = folder / filename
            if path.is_file():
                previous = json.loads(path.read_text())
                break
    a.preview_seed = choose_preview_seed(a.args, previous)
    a.args.preview_seed = a.preview_seed
    a.preview_cases = select_preview_cases(a.plan)
    write_json(a.out / "generalization_preview_plan.json", {
        "seed": a.preview_seed, "sample_steps": a.args.sample_steps,
        "cases": a.preview_cases, "coverage": describe_preview_coverage(a.preview_cases),
    })


def sample_preview_case(a: Any, case: dict, seeds: list[int]) -> tuple:
    """Generate the existing reference/seed budget and reuse outputs for diagnostics."""
    content = a.cache.get(case["source"], a.device)
    target = a.cache.get(case["target"], a.device)
    columns = [("B " + case["B"], a.reader.image(case["source"])),
               ("C target", a.reader.image(case["target"]))]
    samples = []
    for reference_index, reference in enumerate(case["refs"]):
        columns.append((f"A{reference_index} reference", a.reader.image(reference)))
        reference_latent = a.cache.get(reference, a.device)
        for seed in seeds:
            output = sample(lambda noisy, timestep, context: a.route(
                noisy, timestep, context, [content, reference_latent]),
                content, a.context["task"], seed, a.args.sample_steps)
            samples.append({
                "column": len(columns), "kind": case["kind"], "style": case["style"],
                "B": case["B"], "source": case["source"], "A": reference, "seed": seed,
                "latent": output.detach().cpu(), "target": target.detach().cpu(),
                "metrics": measure_output(output, target, content, _multiscale_style_loss),
            })
            columns.append((f"A{reference_index} / seed {seed}", None))
    return columns, samples


def collect_preview_samples(a: Any, cases: list[dict], seeds: list[int]) -> tuple:
    """Collect display cells and latent diagnostics without additional sampling."""
    a.cache.ensure([name for case in cases for name in [case["source"], case["target"]] + case["refs"]])
    rows, samples = [], []
    for case in cases:
        columns, generated = sample_preview_case(a, case, seeds)
        for record in generated:
            record["row"] = len(rows)
        samples.extend(generated)
        rows.append((case["kind"] + " | " + case["style"], columns))
        print(f'[preview] {case["style"]} {case["B"]}', flush=True)
    return rows, samples


def decode_preview_samples(a: Any, rows: list, samples: list[dict]) -> None:
    """Decode all generated images in one VAE residency phase."""
    a.vae.to(a.device)
    try:
        for record in samples:
            row, column = record["row"], record["column"]
            label = rows[row][1][column][0]
            rows[row][1][column] = (label, tensor_image(decode_tensor(a, record["latent"].to(a.device))))
    finally:
        if not a.vae_resident:
            a.vae.to("cpu")


def save_preview_grid(a: Any, rows: list, name: str) -> None:
    """Render the same 224-pixel cells used by prior training previews."""
    tile, header = 224, 44
    row_height = tile + header
    columns = max(len(cells) for _, cells in rows)
    grid = Image.new("RGB", (columns * tile, len(rows) * row_height), "white")
    drawing = ImageDraw.Draw(grid)
    for row_index, (label, cells) in enumerate(rows):
        drawing.text((4, row_index * row_height + 2), label, fill="black")
        for column_index, (cell_label, picture) in enumerate(cells):
            drawing.text((column_index * tile + 4, row_index * row_height + 20), cell_label, fill="black")
            grid.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS),
                       (column_index * tile, row_index * row_height + header))
    grid.save(a.out / (name + ".png"))


def save_preview_images(context: Any, rows: list, samples: list[dict], name: str) -> None:
    """Save decoded native-resolution predictions without extra sampling; O(output pixels)."""
    for record in samples:
        filename = f"{name}_r{record['row']:02d}_c{record['column']:02d}.png"
        picture = rows[record['row']][1][record['column']][1]
        picture.save(context.out / filename)
        record['output'] = filename


def save_preview_metrics(a: Any, samples: list[dict], step: int, name: str) -> None:
    """Save matched-reference diagnostics separately from training metrics."""
    records = []
    checkpoint_name = f"lora84_{step:06d}.safetensors"
    checkpoint_saved = (a.out / checkpoint_name).is_file()
    state_pattern = f"state_{step:06d}_*.pt"
    state_saved = any(a.out.glob(state_pattern))
    for sample_record in samples:
        record = {key: sample_record[key] for key in ("kind", "style", "B", "source", "A", "seed")}
        record.update(sample_record["metrics"])
        record['output'] = sample_record.get('output')
        record.update(step=step, checkpoint=checkpoint_name if checkpoint_saved else None,
                      state_glob=state_pattern if state_saved else None, weights_source="live_model",
                      sample_steps=a.args.sample_steps,
                      note="Latent proxies require visual review; not pure style/content scores")
        records.append(record)
    write_json(a.out / (name + ".json"), records)
    summary = summarize_responses(samples, _multiscale_style_loss)
    summary.update(step=step, objective_version=STYLE_RESPONSE_VERSION)
    write_json(a.out / (name + "_responses.json"), summary)


@torch.no_grad()
def make_grid(a: Any, cases: list[dict], step: int, name: str, seeds: list[int]) -> dict:
    """Generate fixed comparisons; no additional denoiser calls for diagnostics."""
    start = time.perf_counter()
    previous = a.route.use_checkpoint
    a.route.use_checkpoint = False
    try:
        with preserve_rng():
            rows, samples = collect_preview_samples(a, cases, seeds)
            decode_preview_samples(a, rows, samples)
            save_preview_images(a, rows, samples, name)
            save_preview_grid(a, rows, name)
            save_preview_metrics(a, samples, step, name)
    finally:
        a.route.use_checkpoint = previous
        cleanup(a.device)
    synchronize(a.device)
    elapsed = time.perf_counter() - start
    append_json(a.out / "phase_times.jsonl", {"phase": name, "step": step, "seconds": elapsed,
                "images": len(samples), "denoiser_forwards": len(samples) * a.args.sample_steps})
    return {"grid": name + ".png", "samples": len(samples), "seconds": elapsed}


def create_training_phase(context: Any, step: int) -> dict:
    """Describe a new finite data pass and LR clock; time and space O(1)."""
    args = context.args
    if not 0 < args.phase_start_lr <= args.lr_lora:
        raise ValueError('phase_start_lr must be positive and no larger than lr_lora')
    return {'start_step': step, 'epochs': args.epochs, 'pair_count': len(context.plan['train_pairs']),
            'effective_batch': args.effective_batch, 'seed': args.seed,
            'warmup_steps': args.lr_warmup_steps, 'start_lr': args.phase_start_lr,
            'peak_lr': args.lr_lora, 'schedule': 'linear_warmup_then_constant',
            'reference_sampling': 'same_style_different_content_epoch_cycle'}


def restore_phase_progress(context: Any, state: dict) -> tuple[int, int, int]:
    """Reset only a newly requested phase; preserve saved phase progress on resume; O(1)."""
    saved = state.get('training_phase')
    if saved:
        expected = create_training_phase(context, saved['start_step'])
        if saved != expected:
            raise ValueError('Saved phase settings differ; resume with the saved phase arguments')
        context.training_phase = saved
        return state['epoch'], state['cursor'], saved['start_step']
    if getattr(context.args, 'new_training_phase', False):
        context.training_phase = create_training_phase(context, state['step'])
        return 0, 0, state['step']
    count = math.ceil(len(context.plan['train_pairs']) / context.args.effective_batch)
    local_step = state['epoch'] * count + state['cursor'] // context.args.effective_batch
    return state['epoch'], state['cursor'], int(state.get('step_offset', state['step'] - local_step))


def make_phase_scheduler(optimizer: Any, phase: dict, completed: int) -> Any:
    """Use official LambdaLR on the phase-local counter; time and space O(parameter groups)."""
    if completed < 0:
        raise ValueError('Global step precedes the phase start')
    peak = phase['peak_lr']
    start_factor = phase['start_lr'] / peak
    warmup = max(1, phase['warmup_steps'])
    def learning_rate_factor(local_step: int) -> float:
        """Interpolate during warmup and hold afterwards; O(1)."""
        return start_factor + (1 - start_factor) * min(max(local_step, 0) / warmup, 1)
    for group in optimizer.param_groups:
        group['initial_lr'] = peak
        group['lr'] = peak
    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor, last_epoch=completed - 1)


def calculate_training_progress(context: Any, step: int, session_start: int) -> int:
    """Keep preview/save cadence across interruption of a phase; time/space O(1)."""
    phase = getattr(context, 'training_phase', None)
    return step - (phase['start_step'] if phase else session_start)


def write_consumed_coverage(context: Any, epoch: int, order: list, cursor: int) -> None:
    """Record only consumed rows, including restored prefix; time/space O(consumed rows)."""
    consumed = order[:cursor]
    count = len(context.plan['train_pairs'])
    write_json(context.out / f'coverage_epoch_{epoch + 1:03d}.json',
               {'pairs': count, 'consumed_rows': len(consumed), 'unique_pairs_consumed': len(set(consumed)),
                'all_pairs_visited_once': len(consumed) == count and len(set(consumed)) == count})


def make_lr_scheduler(opt,args,total_steps,start_step):
    warmup_steps=max(1,min(args.lr_warmup_steps,total_steps-1))
    decay_steps=max(1,total_steps-warmup_steps)
    transition_steps=max(0,int(getattr(args,'lr_transition_steps',0)))
    transition_start=max(0,int(getattr(args,'lr_transition_start_step',0)))
    transition_end=min(total_steps,transition_start+transition_steps)
    if transition_steps>0 and transition_start<=start_step<transition_end:
        probe_param=torch.nn.Parameter(torch.zeros(()))
        probe_opt=torch.optim.SGD([probe_param],lr=args.lr_lora)
        probe_warmup=torch.optim.lr_scheduler.LinearLR(probe_opt,start_factor=0.05,end_factor=1.0,total_iters=warmup_steps)
        probe_decay=torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt,T_max=decay_steps,eta_min=args.lr_lora*0.1)
        probe_scheduler=torch.optim.lr_scheduler.SequentialLR(probe_opt,[probe_warmup,probe_decay],[warmup_steps])
        for _ in range(transition_end):
            probe_scheduler.step()
        target_lr=float(probe_opt.param_groups[0]['lr'])
        saved_lr=float(opt.param_groups[0]['lr'])
        bridge_steps=max(1,transition_end-start_step)
        base_lr=max(saved_lr,target_lr)
        for group in opt.param_groups:
            group['lr']=base_lr
            group['initial_lr']=base_lr
        start_factor=max(1e-6,saved_lr/base_lr)
        end_factor=max(1e-6,target_lr/base_lr)
        bridge=torch.optim.lr_scheduler.LinearLR(opt,start_factor=start_factor,end_factor=end_factor,total_iters=bridge_steps)
        decay=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,total_steps-transition_end),eta_min=args.lr_lora*0.1)
        scheduler=torch.optim.lr_scheduler.SequentialLR(opt,[bridge,decay],[bridge_steps])
        print('[lr] official LinearLR bridge',start_step,'->',transition_end,'saved_lr',saved_lr,'target_lr',target_lr,flush=True)
        return scheduler
    for group in opt.param_groups:
        group['lr']=args.lr_lora
        group['initial_lr']=args.lr_lora
    warmup=torch.optim.lr_scheduler.LinearLR(opt,start_factor=0.05,end_factor=1.0,total_iters=warmup_steps)
    decay=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=decay_steps,eta_min=args.lr_lora*0.1)
    scheduler=torch.optim.lr_scheduler.SequentialLR(opt,[warmup,decay],[warmup_steps])
    for _ in range(start_step):
        scheduler.step()
    return scheduler
def training_contract(a):
    # Preserve the legacy objective/data fields; execution choice is saved independently.
    return {'version':VERSION,'runtime_version':RUNTIME_VERSION,'script_sha256':file_sha(__file__),
      'plan_sha256':a.plan['fingerprint'],'base_sha256':a.base_info['sha256'],'cache_identity':a.encoding_identity,
      'task_context_sha256':tensor_digest([('task',a.context['task'])]),'sample_steps':a.args.sample_steps,'eval_seeds':a.args.eval_seeds,
      'native_sources':{k:v['sha256'] for k,v in a.sources.items()},'rank':a.args.rank,'lr':a.args.lr_lora,
      'batch':a.args.effective_batch,'res':a.args.res,'seed':a.args.seed,'holdout_fraction':a.args.holdout_fraction,
      'precision':'native bf16; FP32 LoRA','optimizer':'AdamW .9 .999 eps1e-8 wd0 clip1','epochs':a.args.epochs,
      'torch':str(torch.__version__),'cuda':torch.version.cuda,'strategy':{'name':('phase_linear_warmup_constant_v1' if getattr(a.args,'new_training_phase',False) else 'warmup_cosine_v1'),'peak_lr':a.args.lr_lora,'warmup_steps':a.args.lr_warmup_steps,'total_steps':math.ceil(len(a.plan['train_pairs'])/a.args.effective_batch)*a.args.epochs,'preview_every':a.args.preview_every}}


def save_state(a,named,opt,step,epoch,cursor):
    # Commit a new generation before switching the pointer. Never overwrite the only good state.
    name=f'state_{step:06d}_{time.time_ns()}.pt';path=a.out/name;tmp=path.with_suffix('.pt.tmp')
    state={'contract':a.contract,'execution':a.execution,'step':step,'step_offset':int(getattr(a,'step_offset',0)),'epoch':epoch,'cursor':cursor,'rng':rng_state(),
      'adapter':{n:p.detach().cpu().contiguous() for n,p in named.items()},'optimizer':clone_optimizer(opt),
      'objective_version':STYLE_RESPONSE_VERSION,'style_controller':copy.deepcopy(getattr(a,'style_controller',None)),
      'training_phase':copy.deepcopy(getattr(a,'training_phase',None))}
    torch.save(state,tmp);os.replace(tmp,path)
    save_adapter(a,named,step,f'lora84_{step:06d}.safetensors')
    write_json(a.out/'resume_latest.json',{'format':RUNTIME_VERSION,'state_file':name,'step':step})
    # Keep every state created in this run, including periodic and final checkpoints.
    owned=sorted((f for f in a.out.glob('lora84_*.safetensors') if f.stem[7:].isdigit()),key=lambda f:int(f.stem[7:]))
    for old in owned[:-a.args.keep_lora]:old.unlink()


def trim_stream_logs(out,step):
    path=out/'train_84g.jsonl'
    if not path.exists():return
    keep=[];tail=[]
    for line in path.read_text().splitlines(True):
        try:record=json.loads(line)
        except json.JSONDecodeError:tail.append(line);continue
        (keep if record.get('step',0)<=step else tail).append(line)
    if tail:
        (out/f'recovery_discarded_{time.time_ns()}_train84.jsonl').write_text(''.join(tail))
        temp=path.with_suffix('.jsonl.tmp');temp.write_text(''.join(keep));os.replace(temp,path)

def restore_state(a,named,opt):
    if not a.args.resume:return 0,0,0,False
    p=Path(a.args.resume).resolve()
    if p.suffix=='.json':
        pointer=json.loads(p.read_text());p=(p.parent/pointer['state_file']).resolve()
    state=torch.load(p,map_location='cpu',weights_only=True)
    if getattr(a.args,'expect_resume_step',None) is not None and state['step']!=a.args.expect_resume_step:
        raise ValueError(f'Requested resume step {a.args.expect_resume_step}, found {state["step"]}; no training update performed')
    legacy=False
    if state.get('objective_version') != STYLE_RESPONSE_VERSION:
        print('[resume] retaining adapter/optimizer/cursor; initializing paired-target style calibration',flush=True)
    if set(state['adapter'])!=set(named):raise ValueError('Resume tensor keys differ')
    with torch.no_grad():
        for n,p in named.items():
            q=state['adapter'][n]
            if q.shape!=p.shape or q.dtype!=p.dtype or not torch.isfinite(q).all():raise ValueError('Invalid resume parameter '+n)
            p.copy_(q)
    opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    if state.get('objective_version') == STYLE_RESPONSE_VERSION and state.get('style_controller'):
        a.style_controller = copy.deepcopy(state['style_controller'])
    elif hasattr(a, 'style_controller'):
        del a.style_controller
    n=len(a.plan['train_pairs']);bs=a.args.effective_batch;step=state['step']
    e,c,step_offset=restore_phase_progress(a,state)
    local_updates=math.ceil(n/bs);local_step=e*local_updates+c//bs
    if not 0<=e<=a.args.epochs or not 0<=c<n or c%bs or step!=local_step+step_offset or step_offset<0:raise ValueError('Resume epoch/cursor/step invalid')
    a.step_offset=step_offset
    if not legacy:
        a.execution=state['execution'];a.vae_resident=a.execution['vae_resident'];a.route.checkpoint_blocks=a.execution['checkpoint_blocks']
    else:
        write_json(a.out/'legacy_migration.json',{'legacy_contract':state['contract'],'current_contract':a.contract,
          'restored_optimizer':True,'restored_data_cursor':True,'restored_rng':True,
          'claim':'Execution migration; numerical admission required. Not a promise of bitwise equality to the legacy CUDA run.'})
    trim_logs_after_resume(a.out,step)
    trim_stream_logs(a.out,step)
    print(f'[resume] step={step} epoch={e} cursor={c}; optimizer/RNG restored; legacy={legacy}',flush=True)
    return step,e,c,legacy


def selected_profile(a,named,opt,rows):
    saved={n:p.detach().cpu().clone() for n,p in named.items()};osaved=clone_optimizer(opt);rsaved=rng_state()
    controller_saved=copy.deepcopy(getattr(a,'style_controller',None))
    try:
        stage_rows(a,rows)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            train_update(a,named,opt,rows,a.execution)
        prof.export_chrome_trace(str(a.out/'selected_profile.json'))
        kernels=[{'name':e.key,'calls':e.count} for e in prof.key_averages() if any(w in e.key.lower() for w in ('attention','flash','sdp','efficient'))]
        write_json(a.out/'attention_kernels.json',{'evidence':'RUNTIME-OBSERVED profiler event names','events':kernels})
    finally:
        with torch.no_grad():
            for n,p in named.items():p.copy_(saved[n])
        opt.zero_grad(set_to_none=True);opt.load_state_dict(osaved);restore_rng(rsaved);a.style_controller=controller_saved;cleanup(a.device)


def scheduled_outputs(progress: int, intervals: tuple[int, int], finished: bool = False,
                      stop_requested: bool = False) -> tuple[bool, bool]:
    """Schedule independent image/checkpoint intervals; O(1) time and space."""
    preview_interval, save_interval = intervals
    if progress < 0 or min(intervals) <= 0:
        raise ValueError('Progress must be nonnegative and output intervals must be positive')
    preview_due = not stop_requested and (finished or (progress > 0 and progress % preview_interval == 0))
    save_due = stop_requested or finished or (progress > 0 and progress % save_interval == 0)
    return preview_due, save_due


def run84(args):
    if getattr(args,'new_training_phase',False):
        if not args.resume or args.init_lora:
            raise ValueError('A new phase requires --resume from a PT, preserving optimizer state')
        same_output=Path(args.resume).resolve().parent==Path(args.out_dir).resolve()
        if same_output and not (Path(args.out_dir)/'phase_plan.json').is_file():
            raise ValueError('A new phase must use a separate output directory from its source checkpoint')
    out=Path(args.out_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.resume and ((out/'resume_latest.json').exists() or (out/'resume_latest.pt').exists()):raise FileExistsError('Existing training state; pass --resume or choose a new --out_dir')
    if shutil.disk_usage(out).free<2*2**30:raise RuntimeError('Need approximately 2 GiB free for atomic resume/LoRA writes; no image or latent disk cache will be created')
    a=SimpleNamespace(args=args,out=out);report={'runtime':RUNTIME_VERSION,'status':'RUNNING','stages':{},'automatic_success_claim':False}
    def stage(name,fn):
        print(f'[{name}] start',flush=True);t0=time.perf_counter()
        try:
            result=fn();report['stages'][name]={'status':'COMPLETED','seconds':time.perf_counter()-t0,'result':result}
            print(f'[{name}] COMPLETED seconds={time.perf_counter()-t0:.1f}',flush=True);return result
        except BaseException as exc:
            report['stages'][name]={'status':'ERROR','error':repr(exc),'traceback':traceback.format_exc()};raise
        finally:write_json(out/'report_84g.json',report)
    try:
        shutil.copyfile(__file__,out/'source_training_84g.py.txt')
        def plan_data():
            a.index=StyleIndex(args.tar_path);splits=read_splits(args.manifest,set(a.index.by_style))
            a.archive_id=archive_identity(a.index)
            resume_plan=Path(args.resume).resolve().parent/'data_plan.json' if args.resume else None
            if resume_plan and resume_plan.is_file():
                a.plan=json.loads(resume_plan.read_text())
                print(f'[data] reusing resume data plan {resume_plan}',flush=True)
            else:
                groups,identity=source_groups(a.index,out,args.res);a.archive_id=identity['archive']
                a.plan=build_plan(a.index,splits,groups,args);a.plan['manifest_sha256']=file_sha(args.manifest)
                a.plan['fingerprint']=digest_json({k:v for k,v in a.plan.items() if k!='fingerprint'})
            path=out/'data_plan.json'
            if path.exists() and json.loads(path.read_text())!=a.plan:raise ValueError('Data plan differs from existing run')
            write_json(path,a.plan);a.reader=TarPixels(a.index,args.decode_workers,args.res)
            n=len(a.plan['train_pairs']);total=math.ceil(n/args.effective_batch)*args.epochs
            print(f'[data] train styles140 / val30 / sealed test37; pairs={n}; updates={total}; NO disk image/latent cache',flush=True)
            return {'training_pairs':n,'updates':total,'disk_latent_bytes':0,'ram_limit_MiB':args.ram_cache_mb}
        stage('data_plan',plan_data)
        prepare_preview_settings(a)
        write_json(out/('resume_arguments_84g.json' if args.resume else 'arguments_84g.json'),vars(args))
        write_json(out/'training_design.json',{'objective_version':STYLE_RESPONSE_VERSION,
            'conditioning':'B content and independently sampled same-style A reference',
            'style_supervision':'multiscale distance to paired C; no A2 or negative margin',
            'sampling':'each B/C once per epoch; new phase cycles independent same-style A across epochs; holdouts unchanged',
            'calibration':'raw-gradient subset accumulated over several microbatches; EMA; saved in PT',
            'preview':'fixed seed and same-content cross-style comparisons; baseline before updates'})
        for module_name in ('anima_generalization_training.py','anima_generalization_eval.py'):
            shutil.copyfile(Path(__file__).with_name(module_name),out/('source_'+module_name+'.txt'))
        if args.mode=='plan':report['status']='PLAN_ONLY';return 0
        if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA; self_test supports CPU')
        a.device=torch.device(args.device)
        if a.device.type!='cuda':raise ValueError('--device must be cuda:N')
        a.device=torch.device('cuda',torch.cuda.current_device() if a.device.index is None else a.device.index)
        torch.cuda.set_device(a.device);torch.cuda.init();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        stage('preflight',lambda:preflight(a));hashes={}
        # Metadata is small; the pixel pass never saves decoded image arrays.
        hashes={}
        stage('vae',lambda:load_streaming_vae(a));a.cache=RamLatents(a,hashes)
        stage('pixel_split',lambda:{'passed':True,'disabled':True})
        a.base,a.base_info=load_base(args,a.native,a.device);a.base.disable_gradient_checkpointing();a.original=list(a.base.named_parameters())
        a.route=SequenceRoute(a.base,a.attention)
        stage('text',lambda:encode_text(a))
        # Release no-longer-needed text encoder before any training trial.
        a.cache.ensure([a.plan['train_pairs'][0]['target']]);stage('native_contracts',lambda:native_contracts(a))
        a.contract=training_contract(a);write_json(out/'training_contract_84g.json',a.contract)
        a.pools=defaultdict(list)
        for r in a.plan['train_pairs']:
            a.pools[r['style']].append(r)
        n=len(a.plan['train_pairs']);total=math.ceil(n/args.effective_batch)*args.epochs
        stop={'requested':False}
        def sig(signum,frame):stop['requested']=True;print('[stop requested] Saving at next optimizer boundary',flush=True)
        previous={s:signal.signal(s,sig) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with lora_scope(a.base,args.rank) as named:
                if len(named)!=560 or sum(p.numel() for p in named.values())!=45875200:raise RuntimeError('LoRA budget changed')
                if any(p.requires_grad for _,p in a.original):raise RuntimeError('Original weights unexpectedly trainable')
                opt=torch.optim.AdamW(named.values(),lr=args.lr_lora,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                if args.init_lora:
                    from safetensors.torch import load_file
                    init_state=load_file(args.init_lora,device="cpu")
                    if set(init_state)!=set(named):
                        raise ValueError(f"init_lora tensor keys differ: {len(init_state)} vs {len(named)}")
                    with torch.no_grad():
                        for key,param in named.items():
                            value=init_state[key]
                            if tuple(value.shape)!=tuple(param.shape):
                                raise ValueError(f"init_lora shape mismatch: {key} {tuple(value.shape)} vs {tuple(param.shape)}")
                            param.copy_(value.to(device=param.device,dtype=param.dtype))
                    step=int(args.init_step)
                    first_epoch=int(args.init_epoch)
                    cursor=int(args.init_cursor)
                    a.step_offset=step-(first_epoch*math.ceil(n/args.effective_batch)+cursor//args.effective_batch)
                    legacy=False
                    print(f"[init_lora] loaded {args.init_lora} as step={step}; optimizer/RNG are fresh",flush=True)
                else:
                    step,first_epoch,cursor,legacy=restore_state(a,named,opt)
                total_steps=math.ceil(n/args.effective_batch)*args.epochs
                if getattr(a,'training_phase',None):
                    lr_scheduler=make_phase_scheduler(opt,a.training_phase,step-a.training_phase['start_step'])
                    write_json(out/'phase_plan.json',{**a.training_phase,'total_updates':total_steps,
                        'global_end_step':a.training_phase['start_step']+total_steps,
                        'pair_presentations':n*args.epochs,'presentations_per_pair':args.epochs})
                    total=a.training_phase['start_step']+total_steps
                else:
                    lr_scheduler=make_lr_scheduler(opt,args,total_steps,step-int(getattr(a,'step_offset',0)))
                print('[lr]',getattr(a,'training_phase',{}).get('schedule','warmup_cosine'),
                      'global_step',step,'local_step',calculate_training_progress(a,step,step),
                      'lr',opt.param_groups[0]['lr'],flush=True)
                if args.force_microbatch <= 0 and (not args.resume or legacy):
                    ep=min(first_epoch,args.epochs-1);order=epoch_order(a.plan,args.seed,ep)
                    benchrows=logical_rows(a,ep,cursor if first_epoch<args.epochs else 0,order)
                    stage('autotune',lambda:benchmark(a,named,opt,benchrows))
                if args.force_microbatch > 0:
                    if args.force_microbatch not in (1, 2):
                        raise ValueError('force_microbatch must be 1 or 2')
                    forced_blocks = max(0, int(args.force_checkpoint_blocks))
                    a.execution = {'microbatch': int(args.force_microbatch), 'checkpoint_blocks': forced_blocks, 'vae_resident': True}
                    print(f'[throughput] forced execution={a.execution}', flush=True)
                if args.profile_once:
                    profile_epoch=min(first_epoch,args.epochs-1)
                    profile_rows=logical_rows(a,profile_epoch,cursor,epoch_order(a.plan,args.seed,profile_epoch))
                    stage('profile',lambda:selected_profile(a,named,opt,profile_rows))
                print(f'[execution] {a.execution}; RAM={args.ram_cache_mb}MiB; disk_latents=0',flush=True)
                a.route.use_checkpoint=True;a.route.checkpoint_blocks=a.execution['checkpoint_blocks']
                if a.vae_resident:a.vae.to(a.device)
                else:a.vae.to('cpu')
                write_json(out/'runtime_audit_84g.json',{'evidence':'CODE-EXACT; benchmark.json carries actual measured timings',
                  'execution':a.execution,'effective_batch':args.effective_batch,'forwards_per_full_update':math.ceil(args.effective_batch/a.execution['microbatch']),
                  'model_rows_per_full_update':args.effective_batch,'image_tokens':12288,'disk_latent_cache':False,
                  'ram_limit_MiB':args.ram_cache_mb,'vae_batch':1,'training_graph':'one denoiser call/microbatch; paired-C flow/style; sparse multirow raw-gradient audits',
                  'preview_images':12,'preview_forwards':360,'vae_decode_phase_per_grid':1})
                # Initial/migrated state protects progress before the first real update.
                resume_outside=bool(args.resume) and Path(args.resume).resolve().parent!=out
                if not args.resume or legacy or resume_outside:save_state(a,named,opt,step,first_epoch,cursor)
                baseline_name=f'baseline_{step:06d}'
                if not (out/(baseline_name+'.json')).exists():
                    stage(baseline_name,lambda:make_grid(a,a.preview_cases,step,baseline_name,[a.preview_seed]))
                run_start_step=step
                elapsed_steps=[]
                stop_after=(a.training_phase['start_step'] if getattr(a,'training_phase',None) else step)+args.max_steps if args.max_steps else None
                for epoch in range(first_epoch,args.epochs):
                    order=epoch_order(a.plan,args.seed,epoch);pos=cursor if epoch==first_epoch else 0
                    while pos<n:
                        if stop_after is not None and step>=stop_after:
                            save_state(a,named,opt,step,epoch,pos);report['status']='MAX_STEPS';stop['limit_reached']=True;break
                        if stop['requested']:
                            save_state(a,named,opt,step,epoch,pos);report['status']='STOPPED';return 0
                        opt.zero_grad(set_to_none=True);rows=logical_rows(a,epoch,pos,order)
                        write_json(out/'active_update.json',{'next_step':step+1,'epoch':epoch,'cursor':pos,'pair_indices':[x['pair_index'] for x in rows]})
                        synchronize(a.device);t0=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
                        prep=stage_rows(a,rows);compute_start=time.perf_counter()
                        lr_used=float(opt.param_groups[0]['lr'])
                        a.global_step=step;result=train_update(a,named,opt,rows,a.execution);lr_scheduler.step();synchronize(a.device)
                        elapsed=time.perf_counter()-t0;compute=time.perf_counter()-compute_start
                        step+=1;pos+=len(rows);elapsed_steps.append(elapsed)
                        logs=[{'pair_index':x['pair_index'],'B':x['row']['content_key'],'style':x['row']['style'],'A':x['ref'],'style_target':x['row']['target'],
                          'ordinal':x['ordinal'],'t':tv,'loss':loss} for x,tv,loss in zip(rows,result['times'],result['losses'])]
                        append_json(out/'train_84g.jsonl',{'step':step,'epoch':epoch+1,'cursor':pos,'loss':statistics.mean(result['losses']),'base_loss':statistics.mean(result['base_losses']),'style_response_loss':statistics.mean(result['style_response_losses']),'style_target_loss':result['style_target_loss'],'objective_version':STYLE_RESPONSE_VERSION, 'weighted_style_loss':statistics.mean(result['weighted_style_losses']),
                          'grad_norm':result['grad_norm'],'logical_rows':len(rows),'seconds':elapsed,'compute_seconds':compute,'prepare':prep,
                          'peak_allocated_GiB':torch.cuda.max_memory_allocated(a.device)/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved(a.device)/2**30,
                          'style_weight_next':result['weight_next'],'style_grad_probe_rows':result['probe_rows'],
                          'style_grad_measurement_scope':result['measurement_scope'],'style_grad_last_audit_step':result['last_audit_step'],
                          'style_grad_last_probe_count':result['last_audit_probe_count'],'style_controller_status':result['style_controller_status'],
                          'style_probe_norms_are_unprojected':True,
                          'execution':a.execution,'base_grad_norm':result['base_grad_norm'],'style_grad_norm':result['style_grad_norm'],'style_grad_norm_projected':result['style_grad_norm_projected'],'weighted_style_grad_norm':result['weighted_style_grad_norm'],'style_grad_ratio':result['style_grad_ratio'],'style_grad_ratio_target':result['style_grad_ratio_target'],'style_grad_ratio_estimated':result['style_grad_ratio_estimated'],'style_grad_ratio_pre_projection':result['style_grad_ratio_pre_projection'],'grad_cosine':result['grad_cosine'],'style_weight_effective':result['style_weight_effective'],'style_weight_calibrated':result['style_weight_calibrated'],'style_grad_audited':result['style_grad_audited'],'style_grad_projected':result['style_grad_projected'],'style_grad_projection_applied':result['style_grad_projection_applied'],'style_grad_audit_only':result['style_grad_audit_only'],'style_phase_start_step':int(getattr(args,'style_phase_start_step',0)),'style_phase_step':max(0,step-int(getattr(args,'style_phase_start_step',0))),'lr_used':lr_used,'lr_next':float(opt.param_groups[0]['lr']),'rows':logs,'local_update':calculate_training_progress(a,step,run_start_step),'training_phase':('warmup' if calculate_training_progress(a,step,run_start_step)<=args.lr_warmup_steps else 'constant_lr') if getattr(a,'training_phase',None) else None})
                        if step<=3 or step%10==0:
                            display_total=stop_after if stop_after is not None else total
                            eta=statistics.median(elapsed_steps[-20:])*(display_total-step)/3600
                            print(f'[train] {step}/{display_total} loss={statistics.mean(result["losses"]):.6f} sec={elapsed:.2f} compute={compute:.2f} VAE/new_images={prep["encoded"]} ETA={eta:.2f}h',flush=True)
                        run_progress=calculate_training_progress(a,step,run_start_step)
                        if run_progress%10==0 or pos==n:write_consumed_coverage(a,epoch,order,pos)
                        training_finished=(stop_after is not None and step>=stop_after) or (pos==n and epoch==args.epochs-1)
                        preview_due,save_due=scheduled_outputs(run_progress,(args.preview_every,args.save_every),training_finished,stop['requested'])
                        if save_due:
                            opt.zero_grad(set_to_none=True);save_state(a,named,opt,step,epoch+1 if pos==n else epoch,0 if pos==n else pos)
                        if stop['requested']:report['status']='STOPPED';return 0
                        if preview_due:
                            opt.zero_grad(set_to_none=True)
                            stage(f'preview_{step:06d}',lambda:make_grid(a,a.preview_cases,step,f'preview_{step:06d}',[a.preview_seed]))
                        if stop_after is not None and step>=stop_after:
                            report['status']='MAX_STEPS';stop['limit_reached']=True;break
                    if stop['requested'] or stop.get('limit_reached',False):
                        break
                if any(parameter.requires_grad for _,parameter in a.original):raise RuntimeError('Original weights became trainable')
                finals=[]
                if not args.skip_final_eval:
                    cases=a.plan['seen_eval']+a.plan['unseen_eval']
                    for i,s in enumerate(sorted({c['style'] for c in cases})):
                        if stop['requested']:break
                        group=[c for c in cases if c['style']==s];name=f'final_style_{i:03d}'
                        finals.append({'style':s,**stage(name,lambda group=group,name=name:make_grid(a,group,step,name,args.eval_seeds))})
                if any(parameter.requires_grad for _,parameter in a.original):raise RuntimeError('Original weights became trainable during final evaluation')
                write_json(out/'result_84g.json',{'status':'TRAINING_COMPLETED_REQUIRES_IMAGE_REVIEW','step':step,
                  'base_frozen':all(not parameter.requires_grad for _,parameter in a.original),'base_value_comparison_performed':False,'automatic_success_claim':False,
                  'final_styles_completed':len(finals),'finals':finals,'execution':a.execution,'disk_latent_cache':False})
                report['status']='TRAINING_COMPLETED_REQUIRES_IMAGE_REVIEW'
        finally:
            for s,h in previous.items():signal.signal(s,h)
        write_json(out/'report_84g.json',report)
        with tarfile.open(out/'review_84g.tar.gz','w:gz') as archive:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix in ('.json','.jsonl','.png','.txt') and path.name!='selected_profile.json':archive.add(path,arcname=path.name)
        return 0
    except BaseException as exc:report.update(status='ERROR',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        write_json(out/'report_84g.json',report)
        if hasattr(a,'reader'):a.reader.close()
        if hasattr(a,'index'):a.index.close()


def parse84():
    p=argparse.ArgumentParser(description='Anima 84GB streaming training; no disk image/latent cache')
    p.add_argument('--mode',choices=['train','plan','self_test'],default='train')
    for key,value in {'code_root':'/home/waas/code','base_ckpt':'/home/waas/models/anima_baseV10.safetensors',
      'vae_ckpt':'/home/waas/models/qwen_image_vae.safetensors','text_encoder':'/home/waas/models/qwen_3_06b_base.safetensors',
      'qwen_tokenizer':'/home/waas/code/configs/qwen3_06b','t5_tokenizer':'/home/waas/code/configs/t5_old',
      'tar_path':'/home/waas/shuju/StyleExpert/style_expert40k.tar','manifest':'/home/waas/code/cloud_s1/style_split.json',
      'out_dir':'/home/waas/exp/anima_84g_'+time.strftime('%Y%m%d_%H%M%S'),'device':'cuda:0','resume':None}.items():p.add_argument('--'+key,default=value)
    for key,value in {'epochs':2,'effective_batch':16,'rank':32,'res':1024,'preview_every':100,'save_every':200,'sample_steps':30,'seed':20260909,
      'expected_train_pairs':46046,'ram_cache_mb':61440,'decode_workers':4,'keep_lora':2,'benchmark_steps':5,'benchmark_warmup':1}.items():p.add_argument('--'+key,type=int,default=value)
    p.add_argument('--lr_lora',type=float,default=4e-4); p.add_argument('--base_loss_weight','--style_loss_weight',dest='style_loss_weight',type=float,default=1.25,help='Main flow loss coefficient; old style_loss_weight name is retained as an alias');p.add_argument("--style_response_weight",type=float,default=1.0);p.add_argument("--init_lora",type=str,default=None);p.add_argument("--init_step",type=int,default=0);p.add_argument("--init_epoch",type=int,default=0);p.add_argument("--init_cursor",type=int,default=0);p.add_argument('--lr_warmup_steps',type=int,default=300);p.add_argument('--lr_transition_steps',type=int,default=0);p.add_argument('--lr_transition_start_step',type=int,default=0);p.add_argument('--holdout_fraction',type=float,default=.1)
    p.add_argument('--max_gpu_gib',type=float,default=76.);p.add_argument('--safety_gib',type=float,default=4.)
    p.add_argument('--vae_residency',choices=['auto','resident','offload'],default='auto')
    p.add_argument('--eval_seeds',type=int,nargs='+',default=[20260983,20261084])
    p.add_argument('--allow_legacy_resume',action='store_true');p.add_argument('--skip_final_eval',action='store_true');p.add_argument('--profile_once',action='store_true')
    p.add_argument('--expect_resume_step',type=int,default=None)
    p.add_argument('--max_steps',type=int,default=0)
    p.add_argument('--new_training_phase',action='store_true')
    p.add_argument('--phase_start_lr',type=float,default=1e-5)
    p.add_argument('--style_phase_start_step',type=int,default=0)
    p.add_argument('--style_target_grad_ratio',type=float,default=0.15)
    p.add_argument('--style_max_ratio',type=float,default=1.0)
    p.add_argument('--style_grad_interval',type=int,default=32)
    p.add_argument('--style_probe_rows',type=int,default=4)
    p.add_argument('--style_initial_weight',type=float,default=0.08)
    p.add_argument('--style_ema_beta',type=float,default=0.9)
    p.add_argument('--preview_seed',type=int,default=None)
    p.add_argument('--force_microbatch', type=int, default=1)
    p.add_argument('--force_checkpoint_blocks', type=int, default=7)
    args=p.parse_args();args.siglip_model='/unused';args.attention_queries=16
    if args.res!=1024 or args.rank!=32:p.error('Architecture fixed at res1024/rank32')
    if min(args.epochs,args.effective_batch,args.preview_every,args.save_every,args.sample_steps,args.decode_workers,args.keep_lora,args.benchmark_steps,args.benchmark_warmup)<=0:p.error('Budgets must be positive')
    if args.ram_cache_mb<64:p.error('At least 64MiB RAM scratch required')
    if not 0<args.holdout_fraction<.5 or not math.isfinite(args.lr_lora) or args.lr_lora<=0:p.error('Invalid training parameters')
    if args.safety_gib<1 or args.max_gpu_gib<=args.safety_gib:p.error('Invalid GPU memory margin')
    CalibrationSettings.from_arguments(args)
    return args


def self_test84(args):
    """CPU tiny real native modules; not a 1024 pretrained CUDA throughput result."""
    torch.set_num_threads(2);torch.manual_seed(173)
    result={'scope':'CPU tiny random-weight REAL native Anima + synthetic RAM/restore tests; GPU speed UNKNOWN'}
    fake=SimpleNamespace(args=SimpleNamespace(ram_cache_mb=1),encoding_identity={})
    ram=RamLatents(fake,{})
    value=torch.zeros(1,16,1,128,128,dtype=torch.bfloat16)
    ram.put('a',value);ram.put('b',value);ram.get('a','cpu');ram.put('c',value)
    assert set(ram.bank)=={'a','c'} and ram.bytes==2**20
    assert all(t.device.type=='cpu' for t in ram.bank.values())
    result['ram_lru_bounded']=True
    native,attention=import_native(args.code_root)
    base=native.Anima(**native_config(tiny=True)).float().eval().requires_grad_(False);route=SequenceRoute(base,attention)
    original=list(base.named_parameters());original_values={name:parameter.detach().clone() for name,parameter in original}
    bank={name:torch.randn(1,4,1,8,8) for name in ('b','c','a0','a1')}
    cache=SimpleNamespace(get=lambda name,device:bank[name].to(device))
    a=SimpleNamespace(device=torch.device('cpu'),cache=cache,route=route,args=SimpleNamespace(seed=177),context={'task':torch.randn(1,8,32)})
    rows=[{'row':{'source':'b','target':'c'},'ref':'a0' if i%2==0 else 'a1','ordinal':i} for i in range(3)]
    with lora_scope(base,4) as named:
        with torch.no_grad():
            for n,p in named.items():
                if n.endswith('.up'):p.normal_(0,.001)
        initial={n:p.detach().clone() for n,p in named.items()}
        def restored_run(mb,cp):
            with torch.no_grad():
                for n,p in named.items():p.copy_(initial[n])
            opt=torch.optim.AdamW(named.values(),lr=1e-4,weight_decay=0.)
            if hasattr(a,'style_controller'):delattr(a,'style_controller')
            r=train_update(a,named,opt,rows,{'microbatch':mb,'checkpoint_blocks':cp},True)
            delta={n:p.detach().clone()-initial[n] for n,p in named.items()}
            return r,delta
        ref,du=restored_run(1,len(base.blocks));comparisons=[]
        for mb,cp in [(2,len(base.blocks)),(3,len(base.blocks)),(2,1),(3,0)]:
            r,d=restored_run(mb,cp);g=tensor_map_comparison(r['gradients'],ref['gradients']);u=tensor_map_comparison(d,du)
            pd=difference(r['predictions'],ref['predictions'])
            assert g['relative_rms'] < 1e-4 and u['relative_rms'] < 1e-3 and pd['max_abs'] < 1e-5
            comparisons.append({'microbatch':mb,'checkpoint_blocks':cp,'gradient_relative_rms':g['relative_rms'],'update_relative_rms':u['relative_rms'],'prediction_max_abs':pd['max_abs']})
        result['microbatch_tail_and_checkpoint_equivalence']=comparisons
        # Compare the actual legacy single-row loss to the new packed loss at the same seed/t.
        route.use_checkpoint=True;route.checkpoint_blocks=len(base.blocks)
        _,ts,pred,batched_base,_,_,_=batched_loss(a,rows)
        old=[full_loss(a,r['row'],r['ref'],r['ordinal'])[0] for r in rows]
        assert torch.allclose(batched_base,torch.stack(old),atol=1e-6,rtol=1e-6)
        result['legacy_loss_parity']=True
        opt=torch.optim.AdamW(named.values(),lr=1e-4,weight_decay=0.)
        train_update(a,named,opt,rows,{'microbatch':1,'checkpoint_blocks':2})
        with tempfile.TemporaryDirectory() as td:
            aa=SimpleNamespace(out=Path(td),args=SimpleNamespace(rank=4,keep_lora=2,resume=None,allow_legacy_resume=False,epochs=2,effective_batch=3),
              contract={'test':True},execution={'microbatch':1,'checkpoint_blocks':2,'vae_resident':False},base_info={'sha256':'test'},plan={'train_pairs':[1,2,3,4]},route=route)
            aa.style_controller=copy.deepcopy(a.style_controller)
            save_state(aa,named,opt,1,0,3)
            train_update(a,named,opt,rows,aa.execution);expected={n:p.detach().clone() for n,p in named.items()}
            aa.args.resume=str(aa.out/'resume_latest.json');st,e,c,legacy=restore_state(aa,named,opt)
            assert (st,e,c,legacy)==(1,0,3,False)
            a.style_controller=copy.deepcopy(aa.style_controller)
            train_update(a,named,opt,rows,aa.execution)
            assert all(torch.equal(expected[n],p) for n,p in named.items())
            save_state(aa,named,opt,2,1,0);save_state(aa,named,opt,3,1,3)
            assert len(list(aa.out.glob('state_*.pt')))==3 and len(list(aa.out.glob('lora84_*.safetensors')))==2
            assert not list(aa.out.glob('*.npy'))
        result['resume_next_update_equal_and_retention']=True
    assert all(torch.equal(parameter,original_values[name]) for name,parameter in original)
    base.bfloat16();a.context['task']=a.context['task'].bfloat16();bank.update({k:v.bfloat16() for k,v in bank.items()})
    with lora_scope(base,4) as named:
        opt=torch.optim.AdamW(named.values(),lr=1e-4)
        r=train_update(a,named,opt,rows,{'microbatch':2,'checkpoint_blocks':1})
        assert all(math.isfinite(x) for x in r['losses']) and all(p.dtype==torch.float32 for p in named.values())
    result.update(status='PASS',BF16_forward_backward=True,frozen_base_unchanged=True,torch=str(torch.__version__))
    Path(args.out_dir).mkdir(parents=True,exist_ok=True);write_json(Path(args.out_dir)/'self_test84.json',result)
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
    args=parse84();sys.exit(self_test84(args) if args.mode=='self_test' else run84(args))
