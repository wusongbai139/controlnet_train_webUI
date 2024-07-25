import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")
# 调用 get_logger 函数来获取一个名为 __name__ 的日志记录器
# logger 是一个日志记录器对象，用于记录和输出程序的运行信息，如调试信息、错误信息、警告信息等。
logger = get_logger(__name__) 
if is_torch_npu_available(): # 检查当前环境中是否可用NPU（Neural Processing Unit），如果可用，则配置NPU的参数。
    torch.npu.config.allow_internal_format = False

# 训练过程记录 用于在训练过程中和最终验证阶段，使用ControlNet模型生成图像并记录这些图像，以便进行质量检查和调试
def log_validation(vae, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation: # 在非最终验证阶段，函数会解包加速器中的controlnet模型，
        controlnet = accelerator.unwrap_model(controlnet)
        # 并使用预训练的参数和配置创建一个StableDiffusionXLControlNetPipeline管道。
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    else:
        # 在最终验证阶段，直接从输出目录加载controlnet和vae模型，并构建相应的管道
        controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        if args.pretrained_vae_model_name_or_path is not None:
            vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=weight_dtype)
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype
            )

        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []

        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt, image=validation_image, num_inference_steps=20, generator=generator
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs

# 导入和返回一个与预训练模型路径对应的文本编码器类。
# 它的作用是在训练过程中根据提供的模型名称或路径，以及指定的修订版本和子文件夹，
# 从Hugging Face模型库中动态加载文本编码器类，以便用于生成文本嵌入
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    # 加载模型配置。PretrainedConfig.from_pretrained方法从指定路径和修订版本加载模型配置文件。
    # 这一步获取模型的配置数据，包含模型结构和参数等信息。
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    # 获取模型类名
    # 从模型配置中提取模型的类名（通常存储在architectures字段中）
    model_class = text_encoder_config.architectures[0]

    # 根据提取的类名动态导入对应的模型类
    # 如果类名是CLIPTextModel，则导入CLIPTextModel类。
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    # 如果类名是CLIPTextModelWithProjection，则导入CLIPTextModelWithProjection类。
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else: # 如果类名不在支持的范围内，则抛出异常。
        raise ValueError(f"{model_class} is not supported.")

# 生成一个Markdown文件来保存模型参数、标签、模型的目的、训练数据、性能、使用限制和其他重要元数据
def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

# 训练参数上传，解析命令行参数，允许用户自定义训练配置和行为
def parse_args(input_args=None):
    # 使用argparse.ArgumentParser来定义和管理命令行参数。
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument( # 预训练模型的路径或从huggingface.co/models获取的模型标识符。
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument( # 预训练vae模型的路径或从huggingface.co/models获取的模型标识符。
        "--pretrained_vae_model_name_or_path",
        type=str, 
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument( # controlnet模型的类型
        "--model_type",
        type=str, 
        default='SD15',
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str, # 预训练的controlnet模型路径或从huggingface.co/models获取的模型标识符
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant", # 指定预训练模型的变体。使用这个参数来加载不同精度（如fp16或fp32）的预训练模型
        type=str, 
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",  # 用于指定从 Hugging Face 模型库中下载的预训练模型的版本。例如：--revision "v1.0"
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name", # 预训练的tokenizer名称或路径
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir", # 输出目录，模型预测和检查点将写入此目录。
        type=str,
        default="./out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir", # 下载的模型和数据集的存储目录。
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.") # 可重复训练的随机种子
    parser.add_argument(
        "--resolution", # 输入图像的分辨率，训练/验证数据集中的所有图像将调整为此分辨率。
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h", # 包含在SDXL UNet的裁剪坐标嵌入中的高度坐标。
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w", # 包含在SDXL UNet的裁剪坐标嵌入中的宽度坐标。
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument( # 训练数据加载器的每设备批处理大小。
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1) # 训练的总轮数。
    parser.add_argument( # 执行的总训练步骤数。如果提供，将覆盖num_train_epochs
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument( # 每X步保存一次训练状态检查点
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument( # 要存储的最大检查点数量。
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint", # 是否从以前的检查点恢复训练
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps", # 在执行反向/更新之前积累的更新步骤数
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing", # 是否使用梯度检查点以节省内存，代价是较慢的反向传递
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate", # 初始学习率（在潜在的预热期之后）
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument( # 按GPU数量、梯度积累步骤和批量大小缩放学习率。
        "--scale_lr", # True或False
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler", # 要使用的调度器类型。
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument( # 学习率调度器中的预热步骤数。
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument( # 在cosine_with_restarts调度器中的硬重启次数。
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.") # 多项式调度器的幂因子。
    parser.add_argument(  # 是否使用bitsandbytes的8位Adam优化器。
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument( # 数据加载使用的子进程数。
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.") # Adam优化器的beta1参数。
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.") # Adam优化器的beta2参数。
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.") # 要使用的权重衰减。
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer") # Adam优化器的epsilon值。
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.") # 最大梯度范数。
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.") # 是否将模型推送到Hugging Face Model Hub。
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.") # 用于推送到Model Hub的token。
    parser.add_argument(
        "--hub_model_id", # 与本地output_dir保持同步的仓库名称。
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir", # TensorBoard 日志目录。
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32", # 是否允许在Ampere GPU上使用TF32，以加快训练速度。
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to", # 报告结果和日志的平台。支持的平台有"tensorboard"（默认）、"wandb"和"comet_ml"。
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision", # 是否使用混合精度。选择"fp16"或"bf16"（bfloat16）。bfloat16需要PyTorch>=1.10和Nvidia Ampere GPU。
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument( # 是否使用xformers。
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument( 
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument( # 通过将梯度设置为None而不是零来节省更多内存。
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",  # 指定要训练的数据集的名称。这个数据集可以是HuggingFace数据集库中的公开数据集，也可以是用户私有的或本地文件系统中的数据集。
        type=str, # 加载本地数据集：--dataset_name "/path/to/local/dataset"
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name", # 指定数据集的配置名称。某些数据集可能有多个配置（例如，不同的拆分方式、不同的预处理选项等）。
        type=str,                # 如果数据集只有一个配置，可以将这个参数留空。
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir", # 指定包含训练数据的文件夹路径。文件夹内容必须遵循特定的结构
        type=str,           # 文件夹内需要包含一个train.jsonl文件，用于提供图像的标签信息
        default=None,       # 如果指定了--dataset_name参数，则忽略这个参数。
        help=( 
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument( # 数据集中包含目标图像的列。
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument( # 数据集中包含controlnet条件图像的列。
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument( # 数据集中包含字幕或字幕列表的列。
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument( # 出于调试目的或更快的训练，将训练示例的数量截断为此值。只训练前多少张的数量。
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--shuffle_dataset",
        action="store_true",
        help="Whether to shuffle the training dataset. Default is False.",
    )
    parser.add_argument( # 空字符串替换图像提示的比例。默认为0（不替换提示）。
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(       # 在每--validation_steps步时使用一组提示（prompts）进行评估，
        "--validation_prompt", # 并将结果记录到--report_to指定的跟踪系统中
        type=str,              # 用户可以提供一个或多个提示词，模型将在验证过程中使用这些提示词生成图像，并检查模型的生成质量。
        default=None,          # 不提供此参数可以视为不开启验证
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,     # 在每--validation_steps步时使用一组路径指定的图像进行评估，
        default=None, # 并将结果记录到--report_to指定的跟踪系统中
        nargs="+",    # 不提供此参数可以视为不开启验证
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,  # 为每个--validation_image和--validation_prompt对生成的图像数量
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(  # 每经过X步训练后进行一次验证。验证包括多次运行提示词args.validation_prompt，并记录生成的图像。
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=( # 传递给Accelerator.init_trackers的project_name参数
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

# 加载数据集需要上传的参数，需要上传以下参数：
#dataset_name（可选）：数据集的名称，用于从Hugging Face Hub加载数据集。
#dataset_config_name（可选）：数据集的配置名称。
#train_data_dir（可选）：本地数据集目录。
#cache_dir（可选）：缓存目录，用于存储下载的数据集。
#image_column（可选）：数据集中包含图像数据的列名。
#caption_column（可选）：数据集中包含图像标题或标签的列名。
#conditioning_image_column（可选）：数据集中包含控制条件图像的列名。
#max_train_samples（可选）：最大训练样本数，用于调试或加速训练过程。
#seed（可选）：随机种子，用于数据集打乱的可重复性
def get_train_dataset(args, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # 加载数据集 如果提供了dataset_name参数，则从Hugging Face Hub下载并加载指定的数据集。
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        # 如果提供了train_data_dir参数，则从本地目录加载数据集。
        # 在分布式训练环境中，load_dataset函数保证只有一个本地进程可以并发下载数据集，避免重复下载
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # 获取训练数据集的所有列名 Preprocessing the datasets.We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target. 根据用户提供的参数或默认值确定用于图像、标题和控制条件图像的列名
    if args.image_column is None:
        # 如果用户未提供列名，代码会尝试使用默认值并记录日志信息
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        # 如果用户提供的列名在数据集中不存在，则抛出错误
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    # 确保数据预处理步骤在主进程中优先执行
    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        # 并根据max_train_samples参数限制训练样本的数量
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset

# 将一批文本提示（prompts）编码为嵌入向量，
# 这些嵌入向量将作为输入用于Stable Diffusion模型中的生成过程
# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt。
# 参数：
#   prompt_batch：一批文本提示。
#   text_encoders：一个或多个文本编码器模型，用于将输入ID转换为嵌入向量。
#   tokenizers：一个或多个tokenizer，用于将文本提示转换为输入ID。
#   proportion_empty_prompts：替换为空提示的比例。
#   is_train：指示是否处于训练模式，用于确定处理提示列表的方式。
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    # 初始化用于存储提示嵌入和处理后的文本提示的列表
    prompt_embeds_list = []
    captions = []
    # 处理每个文本提示。如果设置了替换空提示的比例（proportion_empty_prompts），
    # 则按比例随机将部分提示替换为空字符串。对于字符串类型的提示，
    # 直接添加到captions列表中；对于列表或数组类型的提示，在训练时随机选取一个，
    # 否则选取第一个
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    # 编码文本提示。使用预训练的tokenizer和text_encoder对处理后的文本提示进行编码。步骤：
    # 使用tokenizer将文本提示转换为输入ID，并设置适当的填充、截断参数。
    # 使用text_encoder对输入ID进行编码，获取提示嵌入（包括隐藏状态）。
    # 从编码结果中提取感兴趣的隐藏状态并进行调整以适应后续处理。
    # 将处理后的提示嵌入添加到prompt_embeds_list中。
    with torch.no_grad(): # 不进行梯度计算。这样可以节省显存并提高计算速度，因为在这个阶段我们只需要进行前向传播（forward pass），不需要反向传播（backward pass）
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer( # 将文本转化为token ids，并将这些ids填充（padding）到固定的最大长度
                captions, # 是一个文本列表，表示一批输入的文本描述
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt", # 返回的是PyTorch张量。
            )
            text_input_ids = text_inputs.input_ids # 包含了每个文本描述的token ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)
    # 将所有的提示嵌入合并成一个张量，并返回处理后的提示嵌入和池化的提示嵌入
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# 准备训练数据集，首先定义了数据预处理的步骤，然后将这些预处理应用到数据集上。
def prepare_train_dataset(dataset, accelerator): # 定义图像转换和预处理
    # 功能：定义用于训练数据的图像预处理管道。
    # 作用：将原始图像调整为指定分辨率，居中裁剪，然后将图像转换为张量，并标准化到范围[-1, 1]。
    # 意义：标准化的图像可以更好地适应神经网络的输入要求，提高训练稳定性和模型收敛速度。
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # 功能：定义用于ControlNet条件图像的预处理管道。
    # 作用：将条件图像调整为指定分辨率，居中裁剪，然后将图像转换为张量。
    # 意义：确保条件图像与训练数据的分辨率一致，以便在训练过程中正确应用条件输入。
    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )
    # 定义数据预处理函数
    # 功能：定义一个用于处理训练数据集的函数。
    # 作用：将训练图像和条件图像转换为RGB格式。应用预定义的图像转换管道（如调整尺寸、裁剪、转换为张量和标准化）。
    #       将处理后的图像存储在数据集的pixel_values和conditioning_pixel_values字段中。
    # 意义：确保训练数据和条件图像都经过相同的预处理步骤，为模型提供一致的输入格式。
    def preprocess_train(examples):

        # 将路径转换为图像对象
        image_paths = [Path(args.train_data_dir) / image_path for image_path in examples[args.image_column]]
        conditioning_image_paths = [Path(args.train_data_dir) / image_path for image_path in examples[args.conditioning_image_column]]

        # Debug: Print the paths to ensure they are correct
        # for image_path in image_paths:
        #     print(f"Loading image: {image_path}")
        # for cond_image_path in conditioning_image_paths:
        #     print(f"Loading conditioning image: {cond_image_path}")

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [image_transforms(image) for image in images]

        conditioning_images = [Image.open(image_path).convert("RGB") for image_path in conditioning_image_paths]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    # 应用预处理到数据集
    # 功能：在主要进程中应用预处理函数到数据集。
    # 作用：使用accelerator.main_process_first()确保在分布式训练环境中，只有主进程首先应用预处理，以避免重复处理。
    # 意义：确保数据预处理步骤在分布式训练环境中高效执行，并且所有进程都能获得相同的预处理数据。
    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)
    return dataset # 将经过预处理的训练数据集返回给调用函数，以便在训练过程中使用

# 它在数据加载器（DataLoader）中用于将一批样本打包成一个批次
def collate_fn(examples):
    # 拼接图像数据. 从每个样本中提取图像数据 pixel_values，
    # 并使用 torch.stack 函数将它们拼接成一个张量。
    # 这种操作确保了所有图像在批处理中保持相同的维度格式。
    # 然后，将这些张量转换为连续内存格式，并将其数据类型转换为浮点数（float）
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    # 拼接条件图像数据。
    # 从每个样本中提取条件图像数据
    # 并将其拼接成一个张量，转换为连续内存格式，并设为浮点数格式
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    # 拼接提示 ID 数据。每个样本中提取提示词嵌入 prompt_embeds，并将其转换为张量，
    # 然后使用 torch.stack 函数将它们拼接成一个批次张量
    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    # 分别从每个样本中提取附加文本嵌入 text_embeds 和时间 ID time_ids，
    # 并将它们转换为张量，然后拼接成批次张量。
    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    # 返回一个包含所有拼接数据的字典
    # 包含了拼接后的图像数据、条件图像数据、提示词 ID 以及 UNet 模型附加条件（包括文本嵌入和时间 ID）。
    # 这些数据将用于模型的输入和训练过程
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }

# 设置训练环境、加载模型和数据、配置优化器和学习率调度器，并启动训练过程
def main(args):
    # 检查是否同时使用了WandB和Hub的token，如果是，抛出错误，因为这可能导致安全问题。
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    # 这段代码设置日志目录，并检查如果使用了MPS（Metal Performance Shaders）并且设置了混合精度为bfloat16，
    # 则抛出错误，因为目前MPS不支持bfloat16。
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # 这段代码初始化Accelerator，用于管理训练过程中的设备和分布式训练。
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 日志目录，并配置accelerator以支持混合精度训练和梯度累积,在主进程上设置适当的日志级别
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 这段代码设置随机种子以保证可重复性。
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # 在主进程上创建输出目录，如果需要的话，还会在Hugging Face Hub上创建一个模型库
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    if args.model_type == 'SDXL':
        # 加载两个预训练的tokenizer
        # Load the tokenizers
        tokenizer_one = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            use_fast=False,
        )
        # 加载噪声调度器、VAE、UNet和ControlNet模型。
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
        # 加载预训练的tokenizers和文本编码器类 Load scheduler and models
        text_encoder_one = text_encoder_cls_one.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        text_encoder_two = text_encoder_cls_two.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
        )
    elif args.model_type == 'SD15':
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, 
                revision=args.revision, 
                use_fast=False)
        elif args.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )
        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision )

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # 检查是否提供了controlnet_model_name_or_path参数。
    # 如果提供了，则从指定路径加载预训练的ControlNet模型权重。如果没有提供，则从UNet初始化ControlNet权重
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # 定义一个辅助函数，用于解包模型以便于访问其原始形式
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 在accelerate版本大于等于0.16.0时，注册自定义的保存和加载钩子，以便更好地序列化模型状态。
    # `accelerate` 0.16.0 will have better support for customized saving
    '''
    对整体模型训练的意义
    一致性和完整性: 通过注册自定义的保存和加载钩子，可以确保模型的保存和加载过程是完整和一致的。这对于分布式训练特别重要，因为所有进程需要共享相同的模型状态。
    灵活性和定制化: 自定义钩子允许根据具体需求调整模型的保存和加载逻辑。例如，可以选择保存哪些模型组件、保存到哪个目录、如何处理不同的模型配置等。
    高效的存储管理: 在训练大规模模型时，定期保存模型状态对于长时间的训练任务非常重要。自定义钩子可以优化保存过程，减少不必要的存储操作，提升整体训练效率。
    恢复训练: 当训练因任何原因中断时，自定义的加载钩子可以确保从上次保存的检查点恢复训练。这样可以避免从头开始重新训练，节省时间和资源。
    兼容性: 这些钩子确保与 accelerate 版本 0.16.0 及以上的兼容性，更好地支持自定义的保存和加载操作。
    '''
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        # 确保模型状态被正确地序列化和保存,检查当前进程是否是主进程，并逐个保存模型权重到指定的输出目录。
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1
        # 确保模型状态被正确地反序列化和加载。依次加载保存的模型权重，并将其恢复到模型中。
        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 冻结VAE、UNet和文本编码器的参数，只训练ControlNet模型
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    if args.model_type == 'SDXL':
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    elif args.model_type == 'SD15':
        text_encoder.requires_grad_(False)
    controlnet.train()

    # 启用NPU闪存注意力机制或xFormers内存高效注意力机制，如果相应的库可用，并启用梯度检查点以节省内存。
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # 检查所有可训练的模型是否为全精度（float32），如果不是，则抛出错误。
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # 启用TF32以在Ampere GPU上加速训练
    # Ampere GPU是NVIDIA推出的一款基于Ampere架构的图形处理单元（GPU）
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    # 根据梯度累积步骤、训练批量大小和进程数调整学习率
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # 使用8-bit Adam优化器以减少内存使用
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation 创建优化器，并设置优化器参数
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # 将VAE、UNet和文本编码器移动到设备上，并根据混合精度设置调整数据类型
    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    if args.model_type == 'SDXL':
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    elif args.model_type == 'SD15':
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    # 定义了一个辅助函数，用于计算文本嵌入和额外的嵌入，以供SD XL UNet使用。
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]

        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds.to(accelerator.device)
        add_text_embeds = add_text_embeds.to(accelerator.device)
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(accelerator.device, dtype=prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

    # 计算所有嵌入以释放文本编码器的内存，然后将其应用于训练数据集
    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    if args.model_type == 'SDXL':
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
    elif args.model_type == 'SD15':
        text_encoders = [text_encoder]
        tokenizers = [tokenizer]
    
    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)

    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    # 准备训练数据集，并创建数据加载器
    train_dataset = prepare_train_dataset(train_dataset, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=args.shuffle_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 配置学习率调度器，并根据训练数据集的大小和设置的训练步数计算相应的步骤数。Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.使用Accelerator准备控制网络、优化器、数据加载器和学习率调度器。
    # accelerator.prepare方法会将优化器和学习率调度器与模型一起准备，使得它们可以正确地处理多GPU环境下的梯度同步和更新。
    # 将数据加载器分发到不同的设备上，确保每个设备都能获得适当的批数据。这在多GPU训练中尤为重要，因为每个GPU需要独立的数据批次进行训练。
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # 重新计算总的训练步数，因为数据加载器的大小可能在accelerator.prepare之后发生了变化。
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs 重新计算训练时间
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化追踪器，并在主进程上存储配置。# We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # 开始训练过程，加载之前保存的模型权重和状态（如果有），并设置初始步数和初始epoch。
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # 这段代码初始化进度条，用于显示训练过程中的进度。
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # 这段代码是训练循环
    image_logs = None # 设置变量：image_logs 用于记录生成的图像日志。
    # 外层循环遍历所有训练轮次（epochs），内层循环遍历每个批次（batches）。
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):             
            with accelerator.accumulate(controlnet): 
                # 图像转换为潜在空间：
                if args.pretrained_vae_model_name_or_path is not None:
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"]
                latents = vae.encode(pixel_values).latent_dist.sample() # 使用VAE编码图像并生成潜在向量
                latents = latents * vae.config.scaling_factor # 根据预训练的VAE模型路径调整潜在向量的数据类型
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents 为每个潜在向量生成与之形状相同的随机噪声。
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image 采样随机时间步
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process) 根据时间步和噪声调度器的噪声幅度将噪声添加到潜在向量，这一步模拟正向扩散过程
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet conditioning. 从批次中提取用于ControlNet条件的图像并转换为指定的数据类型。
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                # 使用ControlNet对噪声潜在向量进行条件处理，生成下采样块和中间块的残差样本。
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                # 使用UNet对噪声潜在向量进行预测，生成噪声残差（noise residual）
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_ids"],
                    added_cond_kwargs=batch["unet_added_conditions"],
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]
                # Get the target for loss depending on the prediction type
                # 根据噪声调度器的预测类型，确定损失的目标。
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # 计算预测的噪声残差与目标之间的均方误差（MSE）损失。
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # 反向传播和优化：使用Accelerator进行反向传播计算梯度，
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    # 如果需要同步梯度，进行梯度裁剪以防止梯度爆炸。
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step() # 更新优化器和学习率调度器。
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none) # 清零优化器的梯度。

            # Checks if the accelerator has performed an optimization step behind the scenes 检查加速器是否在后台执行了优化步骤
            if accelerator.sync_gradients: # sync_gradients 表示是否需要在所有设备上同步梯度
                progress_bar.update(1) # 更新进度条和全局步数。progress_bar.update(1) 表示进度条前进一个单位
                global_step += 1 # 表示当前的全局训练步数增加1。

                # 在DeepSpeed分布式训练或主进程上保存检查点，避免在多个进程上重复保存检查点  DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0: # 检查当前的 global_step 是否达到了保存检查点的间隔步数
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # 管理检查点的数量。如果检查点的数量超过了 checkpoints_total_limit，则删除最旧的检查点以腾出空间保存新的检查点
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        # 将当前训练状态保存为新的检查点。save_path 指定了检查点的保存路径，
                        # accelerator.save_state(save_path) 将当前状态保存到指定路径，并记录日志。
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # 定期进行验证。如果 validation_prompt 被设置且当前 global_step 达到了验证间隔步数（validation_steps），
                    # 则调用 log_validation 函数进行验证，并记录生成的图像日志。
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae=vae,
                            unet=unet,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )
            # 记录当前的损失和学习率，并更新进度条的后缀信息。同时，使用 accelerator.log 函数记录这些日志。
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            # 检查当前的 global_step 是否已经达到了最大训练步数（max_train_steps）。如果是，则跳出循环，结束训练。
            if global_step >= args.max_train_steps:
                break
    # 确保所有进程都完成了当前的操作，并同步等待。这对于分布式训练很重要，确保所有进程在继续下一步操作前都处于相同的状态
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: # 检查当前进程是否是主进程。在分布式训练中，通常只有主进程负责保存模型和进行验证等操作，以避免重复工作
        controlnet = unwrap_model(controlnet) # 使用unwrap_model函数解包模型，确保获得模型的原始形式。这在使用torch.compile等工具包时特别有用，因为它们可能会包装模型对象。
        # 将训练好的ControlNet模型保存到指定的输出目录。save_pretrained方法会将模型权重和配置保存为文件，以便之后加载和使用
        controlnet.save_pretrained(args.output_dir)

        # 进行最后一轮验证。将vae、unet和controlnet设置为None，以便从保存的模型文件中自动加载。
        # 这是为了确保验证使用的模型与保存的一致。log_validation函数运行验证过程，并返回验证结果日志（image_logs）
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                vae=None,
                unet=None,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )
        '''
        检查是否需要将模型推送到Hugging Face Hub。
        如果需要，调用save_model_card函数创建模型卡，并使用upload_folder函数将模型文件上传到指定的模型库。
        上传过程中，忽略以step_和epoch_开头的文件。
        '''
        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    '''
    结束训练过程，执行必要的清理和同步操作。调用end_training确保所有资源都正确释放，所有进程都正常退出。
    '''
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
