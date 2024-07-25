import torch
from torch import Tensor
from safetensors.torch import save_file, load_file

def conv_fp16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.half()

def conv_bf16(t: Tensor):
    if not isinstance(t, Tensor):
        return t
    return t.bfloat16()

def conv_full(t):
    return t

_g_precision_func = {
    "full": conv_full,
    "fp32": conv_full,
    "half": conv_fp16,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}

def convert(path: str, precision: str, conv_type: str, use_safe_tensors: bool):
    ok = {}
    _hf = _g_precision_func[precision]
    
    if path.endswith(".safetensors"):
        m = load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    
    state_dict = m["state_dict"] if "state_dict" in m else m
    
    for k, v in state_dict.items():
        ok[k] = _hf(v)
    
    model_name = ".".join(path.split(".")[:-1])
    save_name = f"{model_name}-{conv_type}-{precision}"
    if use_safe_tensors:
        save_file(ok, save_name + ".safetensors")
    else:
        torch.save({"state_dict": ok}, save_name + ".ckpt")
    
    return f"Converted model saved as {save_name}"