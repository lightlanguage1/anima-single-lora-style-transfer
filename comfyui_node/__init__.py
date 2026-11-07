"""ComfyUI wrapper around the project's native Anima A/B conditioning route."""
from __future__ import annotations

import gc
import importlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGE_SIZE = 1024
MODEL_RANK = 32
RUNTIME_LOCK = threading.RLock()
_RUNTIME: Any = None
_RUNTIME_KEY: tuple[Any, ...] | None = None


def _checked_path(value: str, description: str, directory: bool = False) -> Path:
    """Resolve a required model or source path with a useful error."""
    if not value:
        raise FileNotFoundError(f'{description} is empty')
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f'{description} not found: {value!r}')
    return path


def _file_state(value: str) -> tuple[str, int, int]:
    """Describe a file for cache invalidation without reading its contents."""
    path = Path(value).expanduser().resolve()
    state = path.stat()
    return str(path), state.st_size, state.st_mtime_ns


def _load_modules(runtime_directory: Path, code_root: Path) -> tuple[Any, Any]:
    """Import the bundled project runtime, guarding against unrelated modules."""
    _checked_path(str(runtime_directory / 'evaluate_anima_disentanglement.py'), 'Anima evaluation runtime')
    _checked_path(str(code_root / 'library' / 'anima_models.py'), 'Anima model code')
    for path in (code_root, runtime_directory):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    native = importlib.import_module('anima_native_style_response_from9280_local')
    evaluator = importlib.import_module('evaluate_anima_disentanglement')
    if Path(native.__file__).resolve().parent != runtime_directory or Path(evaluator.__file__).resolve().parent != runtime_directory:
        raise RuntimeError('A different Anima project runtime is already imported; restart ComfyUI before switching runtime folders.')
    return native, evaluator


def _settings(files: tuple[Path, Path, Path], code_root: Path, device: str, steps: int) -> SimpleNamespace:
    """Provide the original inference runtime its explicit local paths."""
    base, vae, text = files
    return SimpleNamespace(device=device, steps=steps, run_dir=None, rank=MODEL_RANK, res=IMAGE_SIZE,
        code_root=str(code_root), base_ckpt=str(base), vae_ckpt=str(vae), text_encoder=str(text),
        qwen_tokenizer=str(code_root / 'configs' / 'qwen3_06b'),
        t5_tokenizer=str(code_root / 'configs' / 't5_old'), tar_path='')


def _release_instance(instance: Any) -> None:
    """Release a fully or partially initialized project runtime."""
    assets = getattr(instance, 'assets', None)
    if assets is not None:
        assets.close()
    scope = getattr(instance, 'adapter_scope', None)
    if scope is not None and hasattr(instance, 'named'):
        scope.__exit__(None, None, None)


def _close_runtime() -> None:
    """Release only the model instance owned by this node."""
    global _RUNTIME, _RUNTIME_KEY
    if _RUNTIME is not None:
        _release_instance(_RUNTIME)
        _RUNTIME = None
        _RUNTIME_KEY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _initialize_runtime(evaluator: Any, native: Any, arguments: SimpleNamespace, checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    """Initialize and clean up the runtime if any loading step fails."""
    instance = evaluator.AnimaTestRuntime.__new__(evaluator.AnimaTestRuntime)
    original_writer = native.write_json
    native.write_json = lambda path, value: None
    try:
        evaluator.AnimaTestRuntime.__init__(instance, arguments, Path(__file__).parent)
        metadata = instance.load(str(checkpoint))
        return instance, metadata
    except BaseException:
        _release_instance(instance)
        del instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        native.write_json = original_writer


def _get_runtime(files: tuple[str, str, str, str], code_root: str, runtime_directory: str,
                 device: str, steps: int, reload_models: int) -> tuple[Any, Any, dict[str, Any]]:
    """Reload when model files, device, runtime folder or reload control change."""
    global _RUNTIME, _RUNTIME_KEY
    paths = tuple(_checked_path(value, label) for value, label in zip(files, ('Anima base', 'VAE', 'text encoder', 'LoRA')))
    code = _checked_path(code_root or str(Path(__file__).parent / 'native_support'), 'project code directory', True)
    runtime = _checked_path(runtime_directory or str(Path(__file__).parent / 'runtime'), 'runtime script directory', True)
    key = tuple(_file_state(str(path)) for path in paths) + (str(code), str(runtime), device, int(reload_models))
    native, evaluator = _load_modules(runtime, code)
    if _RUNTIME is None or _RUNTIME_KEY != key:
        _close_runtime()
        instance, metadata = _initialize_runtime(evaluator, native, _settings(paths[:3], code, device, steps), paths[3])
        _RUNTIME, _RUNTIME_KEY = instance, key
        _RUNTIME.adapter_metadata = metadata
    _RUNTIME.steps = steps
    return _RUNTIME, native, _RUNTIME.adapter_metadata


def _prepare_image(image: torch.Tensor, mode: str) -> np.ndarray:
    """Fit one ComfyUI image to a square and composite alpha on neutral gray."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (1, 3, 4):
        raise ValueError('B and A must each contain exactly one grayscale, RGB or RGBA image')
    pixels = image[0].detach().float().cpu()
    if not torch.isfinite(pixels).all():
        raise ValueError('Input image contains non-finite values')
    pixels = pixels.clamp(0, 1)
    if pixels.shape[-1] == 1:
        pixels = pixels.repeat(1, 1, 3)
    elif pixels.shape[-1] == 4:
        alpha = pixels[..., 3:4]
        pixels = pixels[..., :3] * alpha + (127 / 255) * (1 - alpha)
    array = (pixels * 255).round().to(torch.uint8).numpy()
    picture = Image.fromarray(array, mode='RGB')
    if mode == 'pad':
        picture = ImageOps.pad(picture, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC, color=(127, 127, 127))
    elif mode == 'center_crop':
        picture = ImageOps.fit(picture, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC)
    else:
        raise ValueError(f'Unsupported aspect-ratio mode: {mode}')
    return np.asarray(picture, dtype=np.float32).copy()


def _encode_image(context: Any, native: Any, image: torch.Tensor, mode: str) -> torch.Tensor:
    """Use the project's VAE normalization for one content or style image."""
    pixels = _prepare_image(image, mode)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].to(context.device) / 127.5 - 1
    with native.amp(context.device):
        encoded = context.vae.encode(tensor.to(torch.bfloat16))
    posterior = encoded['latent_dist'] if isinstance(encoded, dict) else encoded.latent_dist
    result = ((posterior.mode().float() - context.mean) / context.std).to(torch.bfloat16)
    if not torch.isfinite(result).all():
        raise FloatingPointError('The VAE produced a non-finite input latent')
    return result


def _sample_images(runtime: Any, native: Any, image_b: torch.Tensor, image_a: torch.Tensor, mode: str, seed: int, steps: int) -> torch.Tensor:
    """Sample target, B, A through the trained route and decode to ComfyUI IMAGE."""
    context = runtime.context
    content = _encode_image(context, native, image_b, mode)
    reference = _encode_image(context, native, image_a, mode)
    predict = lambda noisy, time, text: context.route(noisy, time, text, [content, reference])
    output = native.sample(predict, content, context.context['task'], seed, steps)
    picture = native.tensor_image(native.decode_tensor(context, output))
    return torch.from_numpy(np.array(picture, dtype=np.float32, copy=True))[None] / 255.0


def _validate_device(device: str) -> None:
    """Require an existing CUDA device with BF16 support for this runtime."""
    if not torch.cuda.is_available():
        raise RuntimeError('This native Anima runtime requires an NVIDIA CUDA GPU')
    index = torch.device(device).index
    if index is None or index >= torch.cuda.device_count():
        raise ValueError(f'CUDA device {device} does not exist on this machine')
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f'{device} does not support BF16 required by this Anima runtime')


class AnimaNativeStyleTransfer:
    """Generate with the original single-LoRA three-slot Anima inference route."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {'required': {
            'content_B': ('IMAGE',), 'style_A': ('IMAGE',),
            'base_model': ('STRING', {'default': '', 'multiline': False}),
            'vae_model': ('STRING', {'default': '', 'multiline': False}),
            'text_encoder': ('STRING', {'default': '', 'multiline': False}),
            'lora_file': ('STRING', {'default': '', 'multiline': False}),
            'project_code_directory': ('STRING', {'default': '', 'multiline': False}),
            'runtime_directory': ('STRING', {'default': '', 'multiline': False}),
            'device': (['cuda:0', 'cuda:1'],), 'aspect_ratio': (['pad', 'center_crop'],),
            'reload_models': ('INT', {'default': 0, 'min': 0, 'max': 0x7fffffff}),
            'seed': ('INT', {'default': 20261014, 'min': 0, 'max': 0x7fffffffffffffff}),
            'steps': ('INT', {'default': 30, 'min': 1, 'max': 100}),
        }}

    @classmethod
    def IS_CHANGED(cls, **inputs: Any) -> tuple[Any, ...]:
        """Tell ComfyUI when model contents may have changed at the same path."""
        states = []
        for key in ('base_model', 'vae_model', 'text_encoder', 'lora_file'):
            value = inputs.get(key, '')
            try:
                states.append(_file_state(value))
            except OSError:
                states.append((value, 'missing'))
        return tuple(states)

    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'run_info')
    FUNCTION = 'generate'
    CATEGORY = 'Anima/Research'

    @torch.inference_mode()
    def generate(self, content_B: torch.Tensor, style_A: torch.Tensor, base_model: str, vae_model: str,
                 text_encoder: str, lora_file: str, project_code_directory: str, runtime_directory: str,
                 device: str, aspect_ratio: str, reload_models: int, seed: int, steps: int) -> tuple[torch.Tensor, str]:
        _validate_device(device)
        previous_device = torch.cuda.current_device()
        try:
            with RUNTIME_LOCK:
                files = (base_model, vae_model, text_encoder, lora_file)
                runtime, native, metadata = _get_runtime(files, project_code_directory, runtime_directory, device, steps, reload_models)
                image = _sample_images(runtime, native, content_B, style_A, aspect_ratio, seed, steps)
        finally:
            torch.cuda.set_device(previous_device)
        info = {'checkpoint_step': metadata.get('step'), 'seed': seed, 'steps': steps,
                'resolution': IMAGE_SIZE, 'aspect_ratio': aspect_ratio, 'route': 'target + B + A'}
        return image, json.dumps(info, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {'AnimaNativeStyleTransfer': AnimaNativeStyleTransfer}
NODE_DISPLAY_NAME_MAPPINGS = {'AnimaNativeStyleTransfer': 'Anima · 内容 B + 画风 A（单 LoRA）'}
