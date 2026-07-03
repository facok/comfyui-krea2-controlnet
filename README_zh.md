# ComfyUI Krea2 Control

这是一个用于在 ComfyUI 中运行 Krea2 Control LoRA 的自定义节点。插件保留 ComfyUI 原生 Krea2 推理流程，在采样时使用 LoRA checkpoint 中扩展后的输入投影层，并注入由 VAE 编码得到的控制图 latent。
<img width="4259" height="2044" alt="Krea2t_00048_" src="https://github.com/user-attachments/assets/9b7390fb-420c-4b15-8c5b-88a24969280e" />

<img width="4259" height="2044" alt="Krea2t_00026_" src="https://github.com/user-attachments/assets/429b7373-5716-420a-ba3e-aae261a580f6" />

## 节点

- `Krea2 Control LoRA Loader`：从 `models/loras` 加载 Krea2 Control LoRA，将兼容的 block LoRA 权重应用到 Krea2 模型，准备扩展后的输入投影层，并注册采样时需要的模型 wrapper。
- `Krea2 Control Image Encode`：使用传入的 Krea2/Qwen VAE 将任意 `IMAGE` 控制图编码为 latent。可以直接连接 [`comfyui_controlnet_aux`](https://github.com/Fannovel16/comfyui_controlnet_aux) 的 Depth Anything、Canny、OpenPose、lineart、normal 等预处理结果，也可以使用你自己的控制图。本插件只消费图像输出，不导入、不调用 `comfyui_controlnet_aux` 的代码。
- `Krea2 Control Apply`：将控制 latent 转换到 Krea2 模型使用的 latent 空间，并挂载到已经加载 Control LoRA 的模型上。

## 基本流程
<img width="2018" height="721" alt="8c831c0055122b8090e69e2dd97cbce5" src="https://github.com/user-attachments/assets/1859f3d0-eb56-4729-abc2-8757af77bc34" />
1. 准备控制图。可以用普通 ComfyUI 节点加载图片，也可以连接 [`comfyui_controlnet_aux`](https://github.com/Fannovel16/comfyui_controlnet_aux) 的预处理器输出。
2. 用 `Krea2 Control Image Encode` 编码控制图。保持默认的 `match_latent_size`，并把采样用的 latent 接到该节点的 `latent` 输入。
3. 用 `Krea2 Control LoRA Loader` 加载对应的 Krea2 Control LoRA。
4. 用 `Krea2 Control Apply` 将控制 latent 挂到模型上。
5. 将输出的模型接入 sampler。

`Krea2 Control LoRA Loader` 后必须接 `Krea2 Control Apply`。如果只加载 Control LoRA 但没有挂载控制 latent，采样会直接报错，避免静默运行一个只被部分 patch 的模型。

Block LoRA 权重通过 ComfyUI 的 `ModelPatcher` 应用，因此模型加载、卸载和低显存行为仍然遵循 ComfyUI 原生机制。在 Krea2 diffusion forward 调用期间，图像 token 仍然经过原生 `first` 投影层，因此普通 LoRA 对基础模型的 patch 仍会生效；Control LoRA 只贡献扩展投影中的 control-token 部分。临时投影状态会在调用结束后立即恢复，删除节点后不会把基础 Krea2 生图路径留在 patch 状态。

`match_latent_size` 是默认设置。参考流程会先把控制图缩放到最终生成尺寸，再进行 VAE 编码；这里也按同样的顺序处理。如果你已经用其他 ComfyUI 图像节点提前完成了裁切和缩放，可以改用 `keep_control_image_size`。

## 控制图选项

`Krea2 Control Image Encode` 是通用编码节点，不负责运行 depth、canny、pose 等预处理器。节点里的选项只做轻量图像处理：

- `channel_mode`：保留 RGB 控制图，或转为灰度后复制回 RGB 再送入 VAE。
- `normalize`：`per_image_minmax` 会对每张图单独做 min/max 归一化，适合匹配参考 depth 流程。
- `invert`：翻转 `[0,1]` 控制值。当前处理器的远近、前景背景约定和 LoRA 训练约定相反时再打开。
- `batch_mode`：`independent_images` 会把图像 batch 当作多张独立图片编码，避免 3D Krea2/Qwen VAE 将 batch 解释为视频帧；`video_frames` 则保留 ComfyUI 默认的视频式 VAE 行为。

## Depth LoRA 建议

公开的 depth LoRA 可以先使用以下设置：

- `channel_mode=grayscale`
- `normalize=per_image_minmax`
- `invert=false`

如果 depth 预览里近处是黑色、远处是白色，再将 `invert` 改为 `true`。

公开 depth LoRA 权重地址：[Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet)

其他控制类型，例如 canny、pose、lineart、normal，通常保持：

- `channel_mode=rgb`
- `normalize=none`
- `invert=false`

控制类型由 LoRA checkpoint 决定。Depth LoRA 需要 depth 类控制图；pose、canny、normal 等 LoRA 需要对应类型的预处理图。

## 致谢

感谢 [Krea-2-controlnet](https://github.com/Tanmaypatil123/Krea-2-controlnet) 对 Krea2 Control LoRA 参考推理流程的整理。
感谢 [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet) 发布公开的 depth Control LoRA 权重。
