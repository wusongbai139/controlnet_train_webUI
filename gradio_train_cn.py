import os
import gradio as gr
import subprocess
import webbrowser
from gen_json import generate_json
from convert_model import convert

# 设置MKL使用单线程
os.environ["MKL_THREADING_LAYER"] = "GNU"

def launch_training(model_type, pretrained_model_path, vae='', controlnet_model='', variant='', revision='', 
                    tokenizer_name='', cache_dir='', 
                    train_data_dir='', dataset_name='', dataset_config_name='', image_column='', conditioning_image_column='',
                    caption_column='', max_train_samples=None, resolution=1024, crops_coords_top_left_w=1024, crops_coords_top_left_h=1024, 
                    dataloader_num_workers=0, shuffle=False, proportion_empty_prompts=None, seed=20240724, batch_size=1, 
                    epochs=20, max_train_steps=None, save_every_n_steps=1000, max_models=None, resume_from_checkpoint='',
                    gradient_accumulation_steps=None, gradient_checkpointing=False, lr_scheduler='constant_with_warmup',
                    lr_warmup_steps=1000, lr_num_cycles=1, lr_power=1.0, learning_rate='0.0001', scale_lr=False,
                    use_8bit_adam=False, adam_beta1=0.9, adam_beta2=0.999, adam_weight_decay=1e-2, adam_epsilon=1e-08, 
                    max_grad_norm=1.0, output_dir='./out', logging_dir='', report_to='tensorboard', push_to_hub=False, 
                    hub_token='', hub_model_id='', allow_tf32=False, mixed_precision='no', xformers=False, 
                    npu_flash_attention=False, set_grads_to_none=False, validation_prompt='', validation_image='', 
                    num_validation_images=None, validation_steps=None, tracker_project_name=''):
    command = [
        "accelerate", "launch", "./controlnet_train_15andXL.py",
        "--model_type", model_type,
        "--pretrained_model_name_or_path", pretrained_model_path,
        "--pretrained_vae_model_name_or_path", vae,
        "--controlnet_model_name_or_path", controlnet_model,
        "--variant", variant,
        "--revision", revision,
        "--tokenizer_name", tokenizer_name,
        "--cache_dir", cache_dir,
        "--train_data_dir", train_data_dir,
        "--dataset_name", dataset_name,
        "--dataset_config_name", dataset_config_name,
        "--image_column", image_column,
        "--conditioning_image_column", conditioning_image_column,
        "--caption_column", caption_column,
        "--max_train_samples", str(max_train_samples) if max_train_samples else '',
        "--resolution", str(resolution),
        "--crops_coords_top_left_w", str(crops_coords_top_left_w),
        "--crops_coords_top_left_h", str(crops_coords_top_left_h),
        "--dataloader_num_workers", str(dataloader_num_workers),
        "--shuffle_dataset", str(shuffle),
        "--proportion_empty_prompts", str(proportion_empty_prompts) if proportion_empty_prompts else '',
        "--seed", str(seed),
        "--train_batch_size", str(batch_size),
        "--num_train_epochs", str(epochs),
        "--max_train_steps", str(max_train_steps) if max_train_steps else '',
        "--checkpointing_steps", str(save_every_n_steps),
        "--checkpoints_total_limit", str(max_models) if max_models else '',
        "--resume_from_checkpoint", resume_from_checkpoint,
        "--gradient_accumulation_steps", str(gradient_accumulation_steps) if gradient_accumulation_steps else '',
        "--gradient_checkpointing", str(gradient_checkpointing),
        "--lr_scheduler", lr_scheduler,
        "--lr_warmup_steps", str(lr_warmup_steps),
        "--lr_num_cycles", str(lr_num_cycles),
        "--lr_power", str(lr_power),
        "--learning_rate", learning_rate,
        "--scale_lr", str(scale_lr),
        "--use_8bit_adam", str(use_8bit_adam),
        "--adam_beta1", str(adam_beta1),
        "--adam_beta2", str(adam_beta2),
        "--adam_weight_decay", str(adam_weight_decay),
        "--adam_epsilon", str(adam_epsilon),
        "--max_grad_norm", str(max_grad_norm),
        "--output_dir", output_dir,
        "--logging_dir", logging_dir,
        "--report_to", report_to,
        "--push_to_hub", str(push_to_hub),
        "--hub_token", hub_token,
        "--hub_model_id", hub_model_id,
        "--allow_tf32", str(allow_tf32),
        "--mixed_precision", mixed_precision,
        "--enable_xformers_memory_efficient_attention", str(xformers),
        "--enable_npu_flash_attention", str(npu_flash_attention), 
        "--set_grads_to_none", str(set_grads_to_none),
        "--validation_prompt", validation_prompt,
        "--validation_image", validation_image,
        "--num_validation_images", str(num_validation_images) if num_validation_images else '',
        "--validation_steps", str(validation_steps) if validation_steps else '',
        "--tracker_project_name", tracker_project_name,
    ]
    command = [arg for arg in command if arg != '']  # 去除空参数
    subprocess.run(command, check=True)
    return "训练已启动，参数已经传递。"

# 读取Markdown文件内容
def load_guidance_content():
    with open("lib/params_guidance.md", "r", encoding="utf-8") as file:
        content = file.read()
    return content

# 定义自定义主题
custom_theme = gr.themes.Soft()

model_train = gr.Interface(
    fn=launch_training,
    inputs=[
        gr.Dropdown(choices=["SDXL", "SD15"], label="Model Type", value="SD15"),
        gr.Textbox(label="预训练模型路径或模型标识符(huggingface) Pretrained Model Path or model identifier(huggingface)"),
        gr.Textbox(label="VAE模型路径(非必填) VAE Model Path"),
        gr.Textbox(label="controlnet模型路径(非必填) ControlNet Model Path"),
        gr.Textbox(label="指定预训练模型的变体(非必填) Variant"),
        gr.Textbox(label="下载(huggingface)的预训练模型的版本(非必填) Revision"),
        gr.Textbox(label="预训练的tokenizer名称或路径(非必填) Tokenizer Name"),
        gr.Textbox(label="缓存目录(非必填) Cache Directory"),
        gr.Textbox(label="训练数据集地址 Training Data Directory"),
        gr.Textbox(label="训练的数据集的名称(非必填) Dataset Name"),
        gr.Textbox(label="数据集的配置名称(非必填) Dataset Config Name"),
        gr.Textbox(label="目标图像的列(非必填) Image Column"),
        gr.Textbox(label="条件图像的列(非必填) Conditioning Image Column"),
        gr.Textbox(label="提示词的列(非必填) Caption Column"),
        gr.Textbox(label="只训练数据集的前多少(非必填) Max Train Samples"),
        gr.Number(label="分辨率 Resolution", value=1024),
        gr.Number(label="宽 Crops Coords Top Left W", value=0),
        gr.Number(label="高 Crops Coords Top Left H", value=0),
        gr.Number(label="数据加载使用的子进程数 Dataloader Num Workers", value=0),
        gr.Checkbox(label="随机加载数据 shuffle"),
        gr.Number(label="空提示词几率 Proportion Empty Prompts", value=0),
        gr.Number(label="种子 Seed", value=20240724),
        gr.Number(label="一次训练几张 Train Batch Size", value=1),
        gr.Number(label="总训练轮数 Number of Training Epochs", value=20),
        gr.Number(label="最大训练步数(会覆盖掉总训练轮数,所以非必填) Max Train Steps"),
        gr.Number(label="每几步存一次 Save Every N Steps", value=1000),
        gr.Number(label="总共存几个模型(非必填) Checkpoints Total Limit"),
        gr.Textbox(label="在上个模型基础上开始训练 Resume From Checkpoint"),
        gr.Number(label="梯度累计步数 Gradient Accumulation Steps"),
        gr.Checkbox(label="梯度检查 Gradient Checkpointing"),
        gr.Dropdown(choices=["constant_with_warmup", "cosine_with_restarts", "polynomial", "constant", "linear", "cosine"], label="学习率调度器 LR Scheduler", value="constant_with_warmup"),
        gr.Number(label="学习率预热步数 LR Warmup Steps", value=1000),
        gr.Number(label="学习率重启次数 LR Num Cycles", value=1),
        gr.Number(label="多项式调度器的幂因子 LR Power", value=1.0),
        gr.Textbox(label="学习率 Learning Rate", value='0.0001'),
        gr.Checkbox(label="缩放学习率 Scale LR"),
        gr.Checkbox(label="使用bitsandbytes的8位Adam优化器 Use 8bit Adam"),
        gr.Number(label="Adam Beta1", value=0.9),
        gr.Number(label="Adam Beta2", value=0.999),
        gr.Number(label="Adam Weight Decay", value=1e-2),
        gr.Number(label="Adam Epsilon", value=1e-08),
        gr.Number(label="最大梯度范数 Max Grad Norm", value=1.0),
        gr.Textbox(label="模型输出目录 Output Directory", value='./out'),
        gr.Textbox(label="日志输出目录 Logging Directory"),
        gr.Dropdown(choices=["tensorboard", "wandb", "comet_ml", "all"], label="报告结果和日志的平台 Report To", value="tensorboard"),
        gr.Checkbox(label="将模型推送到huggingfaceModelHub(非必填) Push To Hub"),
        gr.Textbox(label="推送到ModelHub的token(非必填) Hub Token"),
        gr.Textbox(label="与本地output_dir保持同步的仓库名称(非必填) Hub Model ID"),
        gr.Checkbox(label="在Ampere GPU上使用TF32，以加快训练速度(检查自己显卡) Allow Tf32"),
        gr.Dropdown(choices=["no", "fp16", "bf16"], label="混合精度 Mixed Precision", value="no"),
        gr.Checkbox(label="xformers Enable Xformers Memory Efficient Attention"),
        gr.Checkbox(label="Enable NPU Flash Attention"),
        gr.Checkbox(label="Set Grads To None"),
        gr.Textbox(label="测试提示词(非必填) Validation Prompt"),
        gr.Textbox(label="测试图像(非必填) Validation Image"),
        gr.Number(label="每个测试提示词生成图片的数量(非必填) Num Validation Images"),
        gr.Number(label="每经过X步训练后进行一次验证(非必填) Validation Steps"),
        gr.Textbox(label="设置不同的项目名称(非必填) Tracker Project Name")
    ],
    outputs="text",
    title="controlnet_train_webUI",
    submit_btn="开始训练",
    clear_btn="清空参数",
    theme=custom_theme,
    allow_flagging='never',  # 禁用 Flag 按钮
    description="输入各个参数寻来你属于你的controlnet模型"
)

def process_input(conditioning_image_folder, image_folder, text_folder, output_path):
    result = generate_json(conditioning_image_folder, image_folder, text_folder, output_path)
    return result

tools_interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="条件图片文件夹 Conditioning Image Folder"),
        gr.Textbox(label="目标图片文件夹 Image Folder"),
        gr.Textbox(label="提示词文件夹 Text Folder"),
        gr.Textbox(label="输出路径 Output Path"),
    ],
    allow_flagging='never',  # 禁用 Flag 按钮
    outputs="text",
    title="生成JSON文件 Generate JSON from Folders",
    submit_btn="开始生成",
    clear_btn="一键清空",
    description="输入各个文件夹路径和输出路径，然后生成JSON文件"
)

def interface_func(file_path, precision, type, safe_tensors):
    return convert(file_path, precision, type, safe_tensors)

convert_model = gr.Interface(
    fn=interface_func,
    inputs=[
        gr.Textbox(label="Model File Path"),
        gr.Radio(choices=["fp32", "fp16", "bf16"], label="Precision"),
        gr.Radio(choices=["full", "ema-only", "no-ema"], label="Type"),
        gr.Checkbox(label="Use SafeTensors")
    ],
    submit_btn="开始转换",
    clear_btn="清空",
    outputs="text",
    title="模型转换 - Model Converter",
    description="将你的模型转换为各种类型",
    allow_flagging='never',  # 禁用 Flag 按钮
)

# 创建带有侧边栏的主界面
with gr.Blocks(theme=custom_theme) as main_interface:
    with gr.Row():
        with gr.Column(scale=9):
            with gr.Tabs(elem_id="tabs") as tabs:
                with gr.TabItem("训练"):
                    model_train.render()
                with gr.TabItem("参数指导"):
                    gr.Markdown(load_guidance_content())
                with gr.TabItem("JSON文件生成"):
                    tools_interface.render()
                with gr.TabItem("模型转换"):
                    convert_model.render()

main_interface.launch(server_port=8080, share=True) 
