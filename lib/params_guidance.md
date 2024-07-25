<h1 align="center">参数指导 Parameter guidance</h1>

___

### 1. 预训练模型路径或模型标识符(huggingface) - Pretrained Model Path or model identifier(huggingface)

> 将你本地用作预训练的模型的地址放入此处，也可以用huggingface上的模型标识符，如果使用标识符的话模型会自动下载。<br>
> Put the address of your local model used for pre training here, or use the model identifier on Huggingface. If an identifier is used, the model will automatically download。

- **SD15** [https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)
  - 需要下载的文件：
    ```
    stable-diffusion-v1-5
    |-- feature_extractor
    |-- safety_checker
    |-- scheduler
    |-- text_encoder
    |-- tokenizer
    |-- unet
    |-- vae
    ```

- **SDXL：** [https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main)
  - 需要下载的文件：
    ```
    stable-diffusion-xl-base-1.0
    |-- scheduler
    |-- text_encoder
    |-- text_encoder_2
    |-- tokenizer
    |-- tokenizer_2
    |-- unet
    |-- vae
    ```
___

### 2. VAE模型路径 - VAE Model Path

> 将你的VAE模型放在这里，并不是必要选项。但是如果训练SDXL模型需要用修复后的VAE。<br>
> Placing your VAE model here is not a necessary option. But if training the SDXL model requires the use of repaired VAE.

- **SDXL mixVAE：** [https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main)（sdxl_vae.safetensors）

___

### 3. controlnet模型路径 - ControlNet Model Path

> 将已有的controlnet模型路径放进来，或者放从huggingface.co/models获取的模型标识符。不必要填写。<br>
> Put the existing ControlNet model path in, or the model identifier obtained from huggingface. co/models. No need to fill in.
___

### 4. 指定预训练模型的变体 - Variant

> 该参数只在“预训练模型路径或模型标识符”中填写的是模型标识符情况下起作用。<br>
> This parameter only works when the "pre trained model path or model identifier" is filled with the model identifier.
___

### 5. 下载(huggingface)的预训练模型的版本 - Revision

> 该参数只在“预训练模型路径或模型标识符”中填写的是模型标识符情况下起作用。<br>
> This parameter only works when the "pre trained model path or model identifier" is filled with the model identifier.
___

### 6. 缓存目录 - Cache Directory

> 从huggingface上下载的内容存储的文件路径，不填的话存储到默认路径。<br>
> The file path for storing content downloaded from Huggingface, if left blank, will be stored in the default path.
___

### 7. 训练数据集地址 - Training Data Directory

> 本地的训练集文件夹的路径。里面至少包含以下3个内容。<br>
> The path to the local training set folder. It should contain at least three contents. 
1. image文件夹（目标图片 - Target image）；
2. conditioning_image文件夹（放条件图片 - containing conditional images）；
3. train.json（图片的位置对应关系和标签 - captions）。
___

### 8. 训练的数据集的名称 - Dataset Name

> 从huggingface上自动下载的数据集的名字。使用本地数据集的话这里不用填写。<br>
> The name of the dataset automatically downloaded from Huggingface. If using a local dataset, there is no need to fill in here.

___

### 9. 数据集的配置名称 - Dataset Config Name

> 使用本地数据集的话，不用填写这里。<br>
> If using a local dataset, there is no need to fill in here.
___

### 10. 目标图像的列 - Image Column

> 目标图片的键名。一般情况下，可以不写。<br>
> Key name of the target image.In general, it is not necessary to write.
___

### 11. 条件图像的列 - Conditioning Image Column

> 条件图片的键名。一般情况下，可以不写。<br>
> Key name of the conditioning image.In general, it is not necessary to write.
___

### 12. 提示词的列 - Caption Column

> 提示词的键名。一般情况下，可以不写。<br>
> Key name of the Caption.In general, it is not necessary to write.
___

### 13. 只训练数据集的前多少 - Max Train Samples

> 假设数据集中有1000对数据，你在此处填写800，最后只会训练前800对数据。一般不填写此处。<br>
> Assuming there are 1000 pairs of data in the dataset, if you fill in 800 here, only the first 800 pairs of data will be trained in the end. Usually not filled in here.
___

### 14. 数据加载使用的子进程数 - Dataloader Num Workers

> 用于数据加载的子进程数。0表示数据将加载到主进程中。<br>
> Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
___

### 15. shuffle

> 是否对训练数据集进行洗牌。<br>
> Whether to shuffle the training dataset.
___

### 16. 空提示词几率 - Proportion Empty Prompts

> 空字符串替换图像提示的比例。默认为0（不替换提示）。<br>
> Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).
___

### 17. 一次训练几张 - Train Batch Size

> 如果训练XL的controlnet模型，这里建议写1。<br>
> If training XL's ControlNet model, it is recommended to write 1 here.
___

### 18. 最大训练步数 - Max Train Steps

> 这个参数会覆盖掉总训练轮数,所以不是必须填写。<br>
> This parameter will overwrite the total number of training rounds, so it is not mandatory to fill in
___

### 19. 每几步存一次模型 - Save Every N Steps

> 数据集对的数量 * 总训练轮数 = 总步数。 总步数 / 每几步存一次模型 + 1 = 得到模型总数。<br>
> The number of pairs in the dataset * Total number of training rounds = Number of steps. Total number of steps / saving the model every few steps + 1 = Number of models
___

### 20. 总共存几个模型 - Checkpoints Total Limit

> 要存储的最大模型数量。<br>
> Max number of checkpoints to store
___

### 21. 梯度累计步数 - Gradient Accumulation Steps

> 通过累积多个小批次的梯度，可以有效利用显存，模拟更大的批次大小，从而进行训练。但是训练速度会慢，不过总体而言能够加快训练时间。<br>
> By accumulating gradients from multiple small batches, video memory can be effectively utilized to simulate larger batch sizes for training purposes.But the training speed may be slow, but overall it can speed up the training time.
___

### 22. 梯度检查 - Gradient Checkpointing

> 在进行反向传播计算时，减少显存占用，从而允许训练更大、更深的神经网络模型。<br>
> Reducing video memory usage during backpropagation computation allows for training larger and deeper neural network models.
___

### 23. 学习率调度器 - LR Scheduler

> 动态调整学习率的策略。目前支持:<br>
> "constant_with_warmup", "cosine_with_restarts", "polynomial", "constant", "linear", "cosine"。
___

### 24. 学习率预热步数 - LR Warmup Steps

> 逐渐增加学习率，以帮助模型在训练初期更稳定地收敛。<br>
> Gradually increase the learning rate to help the model converge more stably in the early stages of training.
___

### 25. 学习率重启次数 - LR Num Cycles

> 在cosine_with_restarts调度器中的硬重启次数。选择cosine_with_restarts时候，epoch数量要大。<br>
> The number of hard restarts in the cosine with restart scheduler. When choosing cosine with restart, the number of epochs should be large.
___

### 26. 多项式调度器的幂因子 - LR Power

> 控制学习率衰减的速度和曲线形状。当 LR Power=1 时，学习率线性下降；当LR Power>1时，学习率以更快的速度在初期下降，然后逐渐平缓；当 LR Power<1时，学习率初期下降较慢，后期下降较快。<br>
> Control the speed and curve shape of learning rate decay. When LR Power=1, the learning rate decreases linearly; When LR Power>1, the learning rate decreases at a faster rate in the initial stage and gradually plateaus; When LR Power<1, the learning rate decreases slowly in the early stages and rapidly in the later stages.
___

### 27. 学习率 - Learning Rate

> 决定了每次参数更新时步长的大小。<br>
> Decided the size of the step size for each parameter update.
___

### 28. 缩放学习率 - Scale LR

> 在模型训练过程中，不同层或不同参数组可能需要不同的学习率。通过使用学习率缩放，可以为特定层或参数组设置不同的学习率，以便更好地控制它们的更新步长。<br>
> During the model training process, different layers or parameter groups may require different learning rates. By using learning rate scaling, different learning rates can be set for specific layers or parameter groups to better control their update step size.
___

### 29. 使用bitsandbytes的8位Adam优化器 - Use 8bit Adam

> 使用更低精度的计算通常也意味着更少的能量消耗，对于大规模训练任务尤其有利。需要bitsandbytes。<br>
> Using lower precision calculations usually means less energy consumption, which is particularly advantageous for large-scale training tasks.Need bitsandbytes.
___

### 30. Adam Beta1、Adam Beta2

> Beta1默认值0.9。Beta2默认值0.999。提高两个值可以提高模型的训练效果和收敛速度。<br>
> The default value for Beta1 is 0.9. The default value for Beta2 is 0.999. Increasing two values can improve the training effectiveness and convergence speed of the model.
___

### 32. Adam权重衰减 - Adam Weight Decay

> 防止过拟合，提高了模型的泛化能力和稳定性。如果收敛太慢，可以适当增加此值；如果模型震荡或不稳定，可以适当减小此值。<br>
> Preventing overfitting improves the generalization ability and stability of the model. Preventing overfitting improves the generalization ability and stability of the model. If the convergence is too slow, you can increase this value appropriately; If the model oscillates or becomes unstable, this value can be appropriately reduced.
___

### 33. Adam Epsilon

> 防止在计算过程中出现除以零的情况，确保数值稳定性。通常设为1e-8。<br>
> Prevent the occurrence of dividing by zero during the calculation process and ensure numerical stability. Usually set as 1e-8.
___

### 34. 最大梯度范数 - Max Grad Norm

> 防止梯度爆炸，提高训练稳定性和模型的泛化能力。<br>
> Prevent gradient explosion, improve training stability and model generalization ability.
___

### 35. 日志输出目录 - Logging Directory

> 训练日志输出的文件夹路径。不填写的话自动输出到“Output Directory”中。<br>
> The folder path for training log output. If not filled in, it will be automatically output to the 'Output Directory'.
___

### 36. 报告结果和日志的平台 - Report To

> 支持的平台是“tenserboard”（默认）、“wandb”和“comet_ml”。使用“all”向所有集成报告。<br>
> The supported platforms are "tensorboard" (default), "wandb", and "comet_ml". Use 'all' to report to all integrations.
___

### 37. 混合精度 - Mixed Precision

> 利用不同精度（通常是半精度浮点数FP16和单精度浮点数FP32）进行训练的方法。其主要目的是提高训练速度和效率，同时减少内存和显存的使用。<br>
> The method of training using different accuracies (usually half precision floating-point number FP16 and single precision floating-point number FP32). Its main purpose is to improve training speed and efficiency, while reducing the use of memory and video memory.
___

### 37. 使用xformers - Enable Xformers Memory Efficient Attention

> 减少在训练和推理时计算注意力机制所需的内存消耗。<br>
> Reduce the memory consumption required for computing attention mechanisms during training and inference.
___

### 38. Enable NPU Flash Attention

> 提升模型中注意力机制的效率和性能。<br>
> Improve the efficiency and performance of attention mechanisms in the model.
___

### 39. Set Grads To None

> 减少内存占用，提高计算性能。<br>
> Reduce memory usage and improve computing performance.