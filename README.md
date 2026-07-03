# ComfyUI Krea2 Control

Chinese documentation: [README_zh.md](README_zh.md)

Native-style ComfyUI nodes for Krea2 Control LoRA inference. The plugin keeps ComfyUI's built-in Krea2 inference path intact, uses the expanded input projection from the LoRA checkpoint during sampling, and injects a VAE-encoded control latent.
<img width="4259" height="2044" alt="Krea2t_00048_" src="https://github.com/user-attachments/assets/9b7390fb-420c-4b15-8c5b-88a24969280e" />

<img width="4259" height="2044" alt="Krea2t_00026_" src="https://github.com/user-attachments/assets/429b7373-5716-420a-ba3e-aae261a580f6" />

## Nodes

- `Krea2 Control LoRA Loader`: loads a Krea2 Control LoRA from `models/loras`, applies compatible block LoRA weights to the Krea2 model, prepares the expanded input projection, and registers the sampling wrapper.
- `Krea2 Control Image Encode`: encodes any control `IMAGE` with the supplied Krea2/Qwen VAE. It can consume outputs from [`comfyui_controlnet_aux`](https://github.com/Fannovel16/comfyui_controlnet_aux) preprocessors such as Depth Anything, Canny, OpenPose, lineart, and normal maps, but it does not import or call `comfyui_controlnet_aux`.
- `Krea2 Control Apply`: converts the encoded control latent into the Krea2 latent space and attaches it to the model after the Control LoRA has been loaded.

## Basic Workflow
<img width="2018" height="721" alt="8c831c0055122b8090e69e2dd97cbce5" src="https://github.com/user-attachments/assets/1859f3d0-eb56-4729-abc2-8757af77bc34" />

1. Prepare a control image with normal ComfyUI nodes, or connect a [`comfyui_controlnet_aux`](https://github.com/Fannovel16/comfyui_controlnet_aux) preprocessor output.
2. Encode that control image with `Krea2 Control Image Encode` using the Krea2/Qwen image VAE. Keep the default `match_latent_size` and connect the sampler latent to this node's `latent` input.
3. Load the Krea2 Control LoRA with `Krea2 Control LoRA Loader`.
4. Attach the encoded latent with `Krea2 Control Apply`.
5. Send the resulting model to your sampler.

`Krea2 Control Apply` is required after the loader. If the Control LoRA is loaded without an attached control latent, sampling fails instead of silently running a partially patched model.

Block LoRA weights are applied through ComfyUI's `ModelPatcher` so normal model loading, offload, and low-VRAM behavior still apply. During the Krea2 diffusion forward call, image tokens still pass through the native `first` projection so regular LoRA patches on the base model remain active; the Control LoRA contributes only the control-token half of the expanded projection. The temporary projection state is restored immediately afterwards, so removing the node does not leave the base Krea2 path patched.

`match_latent_size` is the default because the reference flow resizes the control image to the final generation size before VAE encoding. Switch to `keep_control_image_size` only if you already resized and cropped the control image elsewhere.

`Krea2 Control Image Encode` is generic and does not run depth, canny, pose, or other preprocessors. Its image options are lightweight tensor operations:

- `channel_mode`: keep RGB controls as-is, or convert to grayscale and repeat back to RGB before VAE encoding.
- `normalize`: `per_image_minmax` normalizes each image independently and is useful for matching the reference depth flow.
- `invert`: flips `[0,1]` control values when the preprocessor convention is reversed from the LoRA's training convention.
- `batch_mode`: `independent_images` keeps image batches as separate samples when using 3D Krea2/Qwen VAEs; `video_frames` preserves ComfyUI's default video-style VAE behavior.

For the public depth LoRA, a good starting point with a Depth Anything output is `channel_mode=grayscale`, `normalize=per_image_minmax`, and `invert=false`. Turn `invert` on only if the depth preview shows near objects as dark instead of white.

Public depth LoRA weights: [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet)

Other LoRA types such as canny, pose, lineart, or normal should usually use `channel_mode=rgb`, `normalize=none`, and `invert=false`.

The control type is determined by the LoRA checkpoint. A depth LoRA needs a depth-like control image; pose, canny, and normal LoRAs need the matching preprocessor image.

The encode node returns normal VAE-space latents. The apply node performs the same Krea2 latent-format normalization that ComfyUI applies to the sampler's main latent before the DiT sees it.

## Acknowledgements

Thanks to [Krea-2-controlnet](https://github.com/Tanmaypatil123/Krea-2-controlnet) for documenting the reference Krea2 control-LoRA inference pipeline.
Thanks to [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet) for providing the public depth Control LoRA weights.
