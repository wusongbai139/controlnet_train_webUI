$model_type = 'SD15'  # controlnet模型的类型 可选：SD15、SDXL。
$pretrained_model = './stable-diffusion-v1-5'  # 底模路径或huggingface模型标识符
$vae = ''  # 预训练vae模型的路径。可以不写。
$controlnet_model = '' # 预训练的controlnet模型路径或huggingface模型标识符。可以不写。
$variant = '' # 指定预训练模型的变体。使用这个参数来加载不同精度（如fp16或fp32）的预训练模型。可以不写
$revision = '' # 用于指定从 Hugging Face 模型库中下载的预训练模型的版本。例如：--revision "v1.0"
$tokenizer_name = '' # 预训练的tokeniz er名称或路径。可以不写。
$cache_dir = '' # 下载的模型和数据集的存储目录

$train_data_dir = 'H:\AIGC\models\controlnet\hallucinate\trainqrcode2' # 包含训练数据的文件夹路径
$dataset_name = '' # 指定要训练的数据集的名称。这个数据集可以是HuggingFace数据集库中的公开数据集，也可以是用户私有的或本地文件系统中的数据集。
$dataset_config_name = '' # 指定数据集的配置名称
$image_column = '' # 数据集中包含目标图像的列
$conditioning_image_column = '' # 数据集中包含controlnet条件图像的列
$caption_column = '' # 数据集中包含字幕或字幕列表的列
$max_train_samples = '' # 只训练数据集的前多少张

$resolution = '1024' # 输入图像的分辨率，训练/验证数据集中的所有图像将调整为此分辨率
$crops_coords_top_left_h = '1024' # 包含在SDXL UNet的裁剪坐标嵌入中的高度坐标
$crops_coords_top_left_w = '1024' # 包含在SDXL UNet的裁剪坐标嵌入中的宽度坐标
$dataloader_num_workers = '1' # 数据加载使用的子进程数
$shuffle = $true  # 随机训练数据集。$true或$false
$proportion_empty_prompts = '' # 空字符串替换图像提示的比例

$seed = '' # 可重复训练的随机种子
$batch_size = '1' # 一步训练几张图片
$epochs = '2' # 训练的总轮数
$max_train_steps = '' # 执行的总训练步骤数。如果提供，将覆盖epochs参数
$save_every_n_steps = '1000' # 每n步保存一次模型
$max_models = '' # 要存储的最大检查点数量
$resume_from_checkpoint = '' # 是否从以前的检查点恢复训练

$gradient_accumulation_steps = '' # 梯度累计步数
$gradient_checkpointing = '' # 梯度检查点

$lr_scheduler = 'cosine_with_restarts' # 调度器，可选 ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
$lr_warmup_steps = ''  # 学习率调度器中的预热步骤数，默认500，调度器是constant_with_warmup时有用
$lr_num_cycles = 1 # 选择cosine_with_restarts调度器时的硬重启次数，默认是1
$lr_power = '' # 多项式调度器的幂因子
$learning_rate = 5e-6 # 初始学习率（在潜在的预热期之后）,默认5e-6
$scale_lr = $false  # 缩放学习率 True或False

$use_8bit_adam = $false  # 是否使用bitsandbytes的8位Adam优化器 $true或$false
$adam_beta1 = '' # Adam优化器的beta1参数
$adam_beta2 = '' # Adam优化器的beta2参数
$adam_weight_decay = '' # 权重衰减
$adam_epsilon = '' # Adam优化器的epsilon值
$max_grad_norm = '' # 最大梯度范数

$output_dir = './out' # 输出目录，模型预测和检查点将存入此目录

$logging_dir = '' # 日志目录保存
$report_to = '' # 报告结果和日志的平台。支持的平台有"tensorboard"（默认）、"wandb"和"comet_ml"。

$push_to_hub = $false  # 否将模型推送到Hugging Face Model Hub  $true或$false
$hub_token = '' # 用于推送到Model Hub的token
$hub_model_id = '' # 与本地output_dir保持同步的仓库名称


$allow_tf32 = $false  # 是否允许在Ampere GPU上使用TF32，以加快训练速度。$true或$false
$mixed_precision = '' # 是否使用混合精度。选择"fp16"或"bf16"（bfloat16）。bfloat16需要PyTorch>=1.10和Nvidia Ampere GPU。
$xformers = $false  # 是否使用xformers。 $true或$false
$enable_npu_flash_attention = $false  # 启用 NPU Flash Attention 功能

$set_grads_to_none = $false  # 通过将梯度设置为None而不是零来节省更多内存。$true或$false

$validation_prompt = '' # 测试提示词
$validation_image = '' # 测试图像
$num_validation_images = '' # 每个测试提示词生成图片的数量
$validation_steps = '' # 每经过X步训练后进行一次验证

$tracker_project_name = '' # 设置不同的项目名称，以便在使用 Accelerate 库时正确初始化跟踪器


# 创建参数数组
$args_list = @()

if ($pretrained_model) {
    $args_list += "--pretrained_model_name_or_path=$pretrained_model"
}
if ($vae) {
    $args_list += "--pretrained_vae_model_name_or_path=$vae"
}
if ($model_type) {
    $args_list += "--model_type=$model_type"
}
if ($controlnet_model) {
    $args_list += "--controlnet_model_name_or_path=$controlnet_model"
}
if ($variant) {
    $args_list += "--variant=$variant"
}
if ($revision) {
    $args_list += "--revision=$revision"
}
if ($tokenizer_name) {
    $args_list += "--tokenizer_name=$tokenizer_name"
}
if ($output_dir) {
    $args_list += "--output_dir=$output_dir"
}
if ($cache_dir) {
    $args_list += "--cache_dir=$cache_dir"
}
if ($seed) {
    $args_list += "--seed=$seed"
}
if ($resolution) {
    $args_list += "--resolution=$resolution"
}
if ($crops_coords_top_left_h) {
    $args_list += "--crops_coords_top_left_h=$crops_coords_top_left_h"
}
if ($crops_coords_top_left_w) {
    $args_list += "--crops_coords_top_left_w=$crops_coords_top_left_w"
}
if ($dataloader_num_workers) {
    $args_list += "--dataloader_num_workers=$dataloader_num_workers"
}
if ($batch_size) {
    $args_list += "--train_batch_size=$batch_size"
}
if ($max_train_steps) {
    $args_list += "--max_train_steps=$max_train_steps"
} elseif ($epochs) {
    $args_list += "--num_train_epochs=$epochs"
}
if ($save_every_n_steps) {
    $args_list += "--checkpointing_steps=$save_every_n_steps"
}
if ($max_models) {
    $args_list += "--checkpoints_total_limit=$max_models"
}
if ($resume_from_checkpoint) {
    $args_list += "--resume_from_checkpoint=$resume_from_checkpoint"
}
if ($gradient_accumulation_steps) {
    $args_list += "--gradient_accumulation_steps=$gradient_accumulation_steps"
}
if ($gradient_checkpointing) {
    $args_list += "--gradient_checkpointing=$gradient_checkpointing"
}
if ($learning_rate) {
    $args_list += "--learning_rate=$learning_rate"
}
if ($scale_lr -eq $true) {
    $args_list += "--scale_lr"
}
if ($lr_scheduler -eq "constant_with_warmup") {
    $args_list += "--lr_scheduler=$lr_scheduler"
    if ($lr_warmup_steps) {
        $args_list += "--lr_warmup_steps=$lr_warmup_steps"
    }
} elseif ($lr_scheduler -eq "cosine_with_restarts") {
    $args_list += "--lr_scheduler=$lr_scheduler"
    if ($lr_num_cycles) {
        $args_list += "--lr_num_cycles=$lr_num_cycles"
    }
} elseif ($lr_scheduler) {
    $args_list += "--lr_scheduler=$lr_scheduler"
}
if ($lr_power) {
    $args_list += "--lr_power=$lr_power"
}
if ($use_8bit_adam -eq $true) {
    $args_list += "--use_8bit_adam"
}
if ($adam_beta1) {
    $args_list += "--adam_beta1=$adam_beta1"
}
if ($adam_beta2) {
    $args_list += "--adam_beta2=$adam_beta2"
}
if ($adam_weight_decay) {
    $args_list += "--adam_weight_decay=$adam_weight_decay"
}
if ($adam_epsilon) {
    $args_list += "--adam_epsilon=$adam_epsilon"
}
if ($max_grad_norm) {
    $args_list += "--max_grad_norm=$max_grad_norm"
}
if ($push_to_hub -eq $true) {
    $args_list += "--push_to_hub"
}
if ($hub_token) {
    $args_list += "--hub_token=$hub_token"
}
if ($hub_model_id) {
    $args_list += "--hub_model_id=$hub_model_id"
}
if ($logging_dir) {
    $args_list += "--logging_dir=$logging_dir"
}
if ($report_to) {
    $args_list += "--report_to=$report_to"
}
if ($allow_tf32 -eq $true) {
    $args_list += "--allow_tf32"
}
if ($mixed_precision) {
    $args_list += "--mixed_precision=$mixed_precision"
}
if ($xformers -eq $true) {
    $args_list += "--enable_xformers_memory_efficient_attention"
}
if ($enable_npu_flash_attention -eq $true) {
    $args_list += "--enable_npu_flash_attention"
}
if ($set_grads_to_none -eq $true) {
    $args_list += "--set_grads_to_none"
}
if ($dataset_name) {
    $args_list += "--dataset_name=$dataset_name"
}
if ($dataset_config_name) {
    $args_list += "--dataset_config_name=$dataset_config_name"
}
if ($train_data_dir) {
    $args_list += "--train_data_dir=$train_data_dir"
}
if ($image_column) {
    $args_list += "--image_column=$image_column"
}
if ($conditioning_image_column) {
    $args_list += "--conditioning_image_column=$conditioning_image_column"
}
if ($caption_column) {
    $args_list += "--caption_column=$caption_column"
}
if ($max_train_samples) {
    $args_list += "--max_train_samples=$max_train_samples"
}
if ($shuffle -eq $true) {
    $args_list += "--shuffle_dataset"
}
if ($proportion_empty_prompts) {
    $args_list += "--proportion_empty_prompts=$proportion_empty_prompts"
}
if ($validation_prompt) {
    $args_list += "--validation_prompt=$validation_prompt"
}
if ($validation_image) {
    $args_list += "--validation_image=$validation_image"
}
if ($num_validation_images) {
    $args_list += "--num_validation_images=$num_validation_images"
}
if ($validation_steps) {
    $args_list += "--validation_steps=$validation_steps"
}
if ($tracker_project_name) {
    $args_list += "--tracker_project_name=$tracker_project_name"
}

Write-Host "当前参数列表: " $args_list

# 运行训练脚本
$command = "accelerate launch ./controlnet_train_15andXL.py $($args_list -join ' ')"
Invoke-Expression $command

Write-Output "Train finished"
Read-Host | Out-Null
