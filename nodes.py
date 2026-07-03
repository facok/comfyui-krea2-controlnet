import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter


CONTROL_LATENT_KEY = "krea2_control_latent"
WRAPPER_KEY = "krea2_control"
EPS = 1e-6


class Krea2ControlInputProjection(nn.Module):
    def __init__(self, weight, bias=None, image_features=None, original_first=None):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("Krea2 control input projection weight must be a 2D tensor.")

        total_features = weight.shape[1]
        if image_features is None:
            if total_features % 2 != 0:
                raise ValueError("Cannot infer Krea2 image/control feature split from odd input width.")
            image_features = total_features // 2
        if image_features <= 0 or image_features >= total_features:
            raise ValueError("Invalid Krea2 image/control feature split.")

        self.image_features = int(image_features)
        self.control_features = int(total_features - image_features)
        self.out_features = int(weight.shape[0])
        self.in_features = int(total_features)
        self.weight = nn.Parameter(weight.detach().cpu().clone(), requires_grad=False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)
        self.control_tokens = None
        object.__setattr__(self, "_original_first", original_first)

    @property
    def original_first(self):
        return object.__getattribute__(self, "_original_first")

    def set_original_first(self, original_first):
        object.__setattr__(self, "_original_first", original_first)

    def forward(self, image_tokens):
        if image_tokens.shape[-1] != self.image_features:
            raise RuntimeError(
                f"Krea2 control projection expected {self.image_features} image features, "
                f"got {image_tokens.shape[-1]}."
            )

        control_tokens = self.control_tokens
        if control_tokens is None:
            original_first = self.original_first
            if original_first is not None:
                return original_first(image_tokens)
            raise RuntimeError("Krea2 control projection was called without control tokens.")
        if control_tokens.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea2 control token count mismatch: image={image_tokens.shape[1]}, "
                f"control={control_tokens.shape[1]}."
            )
        control_tokens = comfy.utils.repeat_to_batch_size(control_tokens, image_tokens.shape[0])
        control_tokens = control_tokens.to(device=image_tokens.device, dtype=image_tokens.dtype)

        original_first = self.original_first
        if original_first is not None:
            image_out = original_first(image_tokens)
            control_weight = self.weight[:, self.image_features:]
            control_weight = comfy.model_management.cast_to_device(control_weight, image_tokens.device, image_tokens.dtype)
            return image_out + F.linear(control_tokens, control_weight, None)

        x = torch.cat((image_tokens, control_tokens), dim=-1)
        weight = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        bias = None
        if self.bias is not None:
            bias = comfy.model_management.cast_to_device(self.bias, x.device, x.dtype)
        return F.linear(x, weight, bias)


def _tensor_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _resize_image(image, width, height, upscale_method="lanczos", crop="center"):
    samples = image[..., :3].clamp(0.0, 1.0).movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _prepare_control_image(image, channel_mode, normalize, invert):
    if image.ndim != 4:
        raise RuntimeError(f"Krea2 control IMAGE must be 4D [B,H,W,C], got shape {tuple(image.shape)}.")
    if image.shape[-1] < 1:
        raise RuntimeError("Krea2 control IMAGE must have at least one channel.")

    image = image.clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    else:
        image = image[..., :3]

    if channel_mode == "grayscale":
        weights = torch.tensor((0.299, 0.587, 0.114), device=image.device, dtype=image.dtype)
        image = (image * weights).sum(dim=-1, keepdim=True).repeat(1, 1, 1, 3)

    if normalize == "per_image_minmax":
        reduce_dims = tuple(range(1, image.ndim))
        image_min = image.amin(dim=reduce_dims, keepdim=True)
        image_max = image.amax(dim=reduce_dims, keepdim=True)
        image = (image - image_min) / (image_max - image_min).clamp_min(EPS)

    if invert:
        image = 1.0 - image

    return image.clamp(0.0, 1.0)


def _encode_control_image(vae, image, batch_mode):
    latent_dim = getattr(vae, "latent_dim", None)
    treats_batch_as_video = latent_dim == 3 and not getattr(vae, "not_video", False)
    if batch_mode == "independent_images" and treats_batch_as_video and image.shape[0] > 1:
        encoded = []
        for i in range(image.shape[0]):
            encoded.append(vae.encode(image[i : i + 1]))
        return torch.cat(encoded, dim=0)
    return vae.encode(image)


def _latent_dict(samples, vae):
    out = {"samples": samples}
    if hasattr(vae, "spacial_compression_encode"):
        out["downscale_ratio_spacial"] = vae.spacial_compression_encode()
    return out


def _find_first_weight_key(state_dict, out_features, in_features):
    preferred = (
        "first.weight",
        "diffusion_model.first.weight",
        "model.diffusion_model.first.weight",
        "transformer.first.weight",
    )
    for key in preferred:
        value = state_dict.get(key)
        if torch.is_tensor(value) and tuple(value.shape) == (out_features, in_features):
            return key

    for key, value in state_dict.items():
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        if tuple(value.shape) != (out_features, in_features):
            continue
        if key.endswith("first.weight") or key.endswith("img_in.weight"):
            return key
    return None


def _find_matching_bias(state_dict, weight_key, out_features):
    candidates = []
    if weight_key.endswith(".weight"):
        candidates.append(weight_key[:-7] + ".bias")
    candidates.extend(
        (
            "first.bias",
            "diffusion_model.first.bias",
            "model.diffusion_model.first.bias",
            "transformer.first.bias",
        )
    )
    for key in candidates:
        value = state_dict.get(key)
        if torch.is_tensor(value) and tuple(value.shape) == (out_features,):
            return value
    return None


def _strip_known_prefixes(base):
    changed = True
    while changed:
        changed = False
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
    return base


def _target_key_from_lora_base(base):
    base = _strip_known_prefixes(base)
    if base.startswith("blocks."):
        return f"diffusion_model.{base}.weight"
    return None


def _lora_pairs(state_dict):
    pair_specs = (
        (".A", ".B"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_down", ".lora_up"),
        ("_lora.down.weight", "_lora.up.weight"),
    )

    seen = set()
    for down_suffix, up_suffix in pair_specs:
        for down_key in state_dict.keys():
            if not down_key.endswith(down_suffix):
                continue
            base = down_key[: -len(down_suffix)]
            up_key = base + up_suffix
            if up_key not in state_dict:
                continue
            pair_id = (down_key, up_key)
            if pair_id in seen:
                continue
            seen.add(pair_id)
            yield base, down_key, up_key


def _build_lora_patches(state_dict, model_state_dict):
    patches = {}
    loaded_keys = set()
    skipped = []

    for base, down_key, up_key in _lora_pairs(state_dict):
        target_key = _target_key_from_lora_base(base)
        if target_key is None or target_key not in model_state_dict:
            continue

        down = state_dict[down_key]
        up = state_dict[up_key]
        target_shape = tuple(model_state_dict[target_key].shape)
        if not (torch.is_tensor(down) and torch.is_tensor(up) and down.ndim == 2 and up.ndim == 2):
            skipped.append((down_key, up_key, "not 2D tensors"))
            continue

        out_features, in_features = target_shape[0], target_shape[1]
        if up.shape[0] == out_features and down.shape[1] == in_features and up.shape[1] == down.shape[0]:
            rank = down.shape[0]
        elif down.shape[0] == in_features and up.shape[1] == out_features and down.shape[1] == up.shape[0]:
            down = down.t().contiguous()
            up = up.t().contiguous()
            rank = down.shape[0]
        else:
            skipped.append((down_key, up_key, f"shape does not match {target_key}"))
            continue

        alpha_key = None
        alpha = rank
        for suffix in (".alpha", ".network_alpha", ".scale"):
            candidate = base + suffix
            if candidate in state_dict:
                alpha_key = candidate
                alpha = _tensor_scalar(state_dict[candidate])
                break

        keys = {down_key, up_key}
        if alpha_key is not None:
            keys.add(alpha_key)
        patches[target_key] = LoRAAdapter(keys, (up, down, alpha, None, None, None))
        loaded_keys.update(keys)

    if skipped:
        logging.info("Krea2 control skipped %d LoRA tensor pairs with incompatible shapes.", len(skipped))
    return patches, loaded_keys


def _get_first_module(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model.first")
    except Exception as exc:
        raise RuntimeError("The supplied MODEL does not look like a native ComfyUI Krea2 model.") from exc


def _first_shape(first):
    if isinstance(first, Krea2ControlInputProjection):
        return first.out_features, first.image_features, first.control_features
    weight = getattr(first, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise RuntimeError("Krea2 first projection does not expose a 2D weight tensor.")
    return int(weight.shape[0]), int(weight.shape[1]), int(weight.shape[1])


def _make_control_projection(model_patcher, state_dict):
    first = _get_first_module(model_patcher)
    out_features, image_features, control_features = _first_shape(first)
    expected_in = image_features + control_features
    weight_key = _find_first_weight_key(state_dict, out_features, expected_in)
    if weight_key is None:
        raise RuntimeError(
            f"Could not find expanded Krea2 first projection weight with shape "
            f"({out_features}, {expected_in}) in the selected LoRA file."
        )

    bias = _find_matching_bias(state_dict, weight_key, out_features)
    if bias is None and hasattr(first, "bias") and torch.is_tensor(first.bias):
        bias = first.bias.detach()

    if isinstance(first, Krea2ControlInputProjection):
        original_first = first.original_first
    else:
        original_first = first

    return Krea2ControlInputProjection(
        state_dict[weight_key],
        bias=bias,
        image_features=image_features,
        original_first=original_first,
    )


def _clean_original_first(first):
    if isinstance(first, Krea2ControlInputProjection) and first.original_first is not None:
        return first.original_first
    return first


def _flatten_temporal_if_needed(control_latent):
    if control_latent.ndim == 4:
        return control_latent
    if control_latent.ndim == 5:
        b, c, t, h, w = control_latent.shape
        return control_latent.reshape(b * t, c, h, w)
    raise RuntimeError(f"Krea2 control latent must be 4D or 5D, got shape {tuple(control_latent.shape)}.")


def _expected_latent_channels(model_patcher):
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception:
        return None
    return getattr(latent_format, "latent_channels", None)


def _process_control_latent_for_model(model_patcher, control_latent):
    if control_latent.ndim not in (4, 5):
        raise RuntimeError(f"Krea2 control latent must be 4D or 5D, got shape {tuple(control_latent.shape)}.")

    expected_channels = _expected_latent_channels(model_patcher)
    if expected_channels is not None and control_latent.shape[1] != expected_channels:
        raise RuntimeError(
            f"Krea2 control latent has {control_latent.shape[1]} channels, "
            f"but the selected model expects {expected_channels}. Use the Krea2/Qwen image VAE."
        )

    processed = control_latent
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception:
        latent_format = None

    added_time_dim = False
    if latent_format is not None and getattr(latent_format, "latent_dimensions", 2) == 3 and processed.ndim == 4:
        processed = processed.unsqueeze(2)
        added_time_dim = True

    if hasattr(model_patcher.model, "process_latent_in"):
        processed = model_patcher.model.process_latent_in(processed)

    if added_time_dim and processed.ndim == 5 and processed.shape[2] == 1:
        processed = processed[:, :, 0]
    return processed


def _control_tokens_from_latent(control_latent, x, patch, expected_features):
    if x.ndim == 5:
        target_batch = x.shape[0] * x.shape[2]
    elif x.ndim == 4:
        target_batch = x.shape[0]
    else:
        raise RuntimeError(f"Krea2 input latent must be 4D or 5D, got shape {tuple(x.shape)}.")

    control = _flatten_temporal_if_needed(control_latent)
    control = comfy.utils.repeat_to_batch_size(control, target_batch)
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)

    target_h, target_w = x.shape[-2], x.shape[-1]
    if control.shape[-2:] != (target_h, target_w):
        control = comfy.utils.common_upscale(control, target_w, target_h, "bilinear", "disabled")

    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch, patch))
    b, c, h, w = control.shape
    if h % patch != 0 or w % patch != 0:
        raise RuntimeError("Krea2 control latent padding failed to align to patch size.")

    features = c * patch * patch
    if features != expected_features:
        raise RuntimeError(
            f"Krea2 control latent produces {features} token features, "
            f"but the projection expects {expected_features}. Check that you encoded with the Krea2/Qwen image VAE."
        )

    control = control.reshape(b, c, h // patch, patch, w // patch, patch)
    control = control.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch) * (w // patch), features)
    return control


def _get_transformer_options_from_forward(args, kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    if transformer_options is None and len(args) >= 5 and isinstance(args[4], dict):
        transformer_options = args[4]
    if transformer_options is None and len(args) > 0 and isinstance(args[-1], dict):
        transformer_options = args[-1]
    return transformer_options


def _restore_control_projection(diffusion_model, control_projection):
    control_projection.control_tokens = None
    original_first = control_projection.original_first
    if original_first is not None and getattr(diffusion_model, "first", None) is control_projection:
        diffusion_model.first = original_first


def _make_control_projection_injection(control_projection):
    def inject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is None:
            return
        current_first = _clean_original_first(getattr(diffusion_model, "first", None))
        if current_first is not None and current_first is not control_projection:
            control_projection.set_original_first(current_first)
            diffusion_model.first = current_first
        control_projection.control_tokens = None

    def eject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is not None:
            _restore_control_projection(diffusion_model, control_projection)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _restore_control_projection_callback(model_patcher, *args):
    attachment = model_patcher.get_attachment(WRAPPER_KEY)
    if not isinstance(attachment, dict):
        return
    control_projection = attachment.get("control_projection")
    if not isinstance(control_projection, Krea2ControlInputProjection):
        return
    diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
    if diffusion_model is None:
        return
    _restore_control_projection(diffusion_model, control_projection)


def _krea2_control_wrapper(control_projection):
    def wrapper(executor, *args, **kwargs):
        return krea2_control_wrapper(executor, control_projection, *args, **kwargs)

    return wrapper


def krea2_control_wrapper(executor, control_projection, *args, **kwargs):
    transformer_options = _get_transformer_options_from_forward(args, kwargs)
    if not isinstance(transformer_options, dict):
        raise RuntimeError("Krea2 Control LoRA could not find transformer_options during sampling.")

    diffusion_model = executor.class_obj
    control_latent = transformer_options.get(CONTROL_LATENT_KEY)
    if control_latent is None:
        _restore_control_projection(diffusion_model, control_projection)
        raise RuntimeError(
            "Krea2 Control LoRA is loaded, but no control latent is attached. "
            "Connect Krea2 Control Apply after Krea2 Control LoRA Loader, or remove the loader."
        )

    if not isinstance(control_projection, Krea2ControlInputProjection):
        raise RuntimeError("Krea2 Control LoRA input projection is not installed. Reload the base model and loader.")

    x = args[0]
    previous_first = getattr(diffusion_model, "first", None)
    previous_tokens = control_projection.control_tokens
    try:
        control_tokens = _control_tokens_from_latent(
            control_latent,
            x,
            diffusion_model.patch,
            control_projection.control_features,
        )
        control_projection.control_tokens = control_tokens
        if getattr(diffusion_model, "first", None) is not control_projection:
            diffusion_model.first = control_projection
        return executor(*args, **kwargs)
    finally:
        control_projection.control_tokens = previous_tokens
        if getattr(diffusion_model, "first", None) is control_projection:
            original_first = control_projection.original_first
            diffusion_model.first = original_first if original_first is not None else previous_first


class Krea2ControlLoRALoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "Krea2/control"

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        if model.get_attachment(WRAPPER_KEY) is not None:
            raise RuntimeError("Krea2 Control LoRA is already loaded on this MODEL. Use only one loader per model path.")

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state_dict = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                state_dict = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if state_dict is None:
            state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, state_dict)

        new_model = model.clone()
        control_projection = _make_control_projection(new_model, state_dict)
        lora_patches, loaded_keys = _build_lora_patches(state_dict, new_model.model.state_dict())
        if not lora_patches:
            raise RuntimeError("No compatible Krea2 control LoRA block weights were found in the selected file.")

        patched_keys = new_model.add_patches(lora_patches, strength_patch=strength, strength_model=1.0)
        if not patched_keys:
            raise RuntimeError("The selected MODEL did not accept any Krea2 control LoRA patches.")

        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            _krea2_control_wrapper(control_projection),
        )
        new_model.set_injections(WRAPPER_KEY, _make_control_projection_injection(control_projection))
        new_model.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_DETACH,
            WRAPPER_KEY,
            _restore_control_projection_callback,
        )
        new_model.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            WRAPPER_KEY,
            _restore_control_projection_callback,
        )
        new_model.set_attachments(
            WRAPPER_KEY,
            {
                "lora_name": lora_name,
                "strength": strength,
                "loaded_lora_keys": len(loaded_keys),
                "patched_model_keys": len(patched_keys),
                "control_projection": control_projection,
            },
        )
        return (new_model,)


class Krea2ControlApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "control_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "Krea2/control"

    def apply(self, model, control_latent):
        if "samples" not in control_latent:
            raise RuntimeError("control_latent is missing LATENT['samples'].")

        samples = control_latent["samples"]
        if not torch.is_tensor(samples):
            raise RuntimeError("control_latent['samples'] must be a tensor.")

        new_model = model.clone()
        if new_model.get_attachment(WRAPPER_KEY) is None:
            raise RuntimeError("Krea2 Control Apply must receive the MODEL output from Krea2 Control LoRA Loader.")

        samples = _process_control_latent_for_model(new_model, samples)
        transformer_options = new_model.model_options.setdefault("transformer_options", {})
        transformer_options[CONTROL_LATENT_KEY] = samples
        return (new_model,)


class Krea2ControlImageEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_image": ("IMAGE",),
                "vae": ("VAE",),
                "resize": (
                    ["keep_control_image_size", "match_latent_size"],
                    {"default": "match_latent_size"},
                ),
                "upscale_method": (["lanczos", "bicubic", "bilinear", "area", "nearest-exact"], {"default": "lanczos"}),
                "crop": (["center", "disabled"], {"default": "center"}),
                "channel_mode": (["rgb", "grayscale"], {"default": "rgb"}),
                "normalize": (["none", "per_image_minmax"], {"default": "none"}),
                "invert": ("BOOLEAN", {"default": False}),
                "batch_mode": (["independent_images", "video_frames"], {"default": "independent_images"}),
            },
            "optional": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("control_latent", "encoded_control_image")
    FUNCTION = "encode"
    CATEGORY = "Krea2/control"

    def encode(self, control_image, vae, resize, upscale_method, crop, channel_mode, normalize, invert, batch_mode, latent=None):
        image = _prepare_control_image(control_image, "rgb", "none", False)
        if resize == "match_latent_size":
            if latent is None or "samples" not in latent:
                raise RuntimeError("Krea2 Control Image Encode needs a LATENT input when resize is match_latent_size.")
            compression = vae.spacial_compression_encode() if hasattr(vae, "spacial_compression_encode") else 8
            target_height = int(latent["samples"].shape[-2] * compression)
            target_width = int(latent["samples"].shape[-1] * compression)
            image = _resize_image(image, target_width, target_height, upscale_method, crop)

        image = _prepare_control_image(image, channel_mode, normalize, invert)
        samples = _encode_control_image(vae, image, batch_mode)
        return (_latent_dict(samples, vae), image)


NODE_CLASS_MAPPINGS = {
    "Krea2ControlLoRALoader": Krea2ControlLoRALoader,
    "Krea2ControlApply": Krea2ControlApply,
    "Krea2ControlImageEncode": Krea2ControlImageEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2ControlLoRALoader": "Krea2 Control LoRA Loader",
    "Krea2ControlApply": "Krea2 Control Apply",
    "Krea2ControlImageEncode": "Krea2 Control Image Encode",
}
