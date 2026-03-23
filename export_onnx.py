import torch
import numpy as np
from numpy.core.multiarray import _reconstruct
torch.serialization.add_safe_globals([_reconstruct])

# 加载模型
ckpt = torch.load("runs/train/exp23/weights/best.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float()
model.eval()

# 正确导出 YOLOv5 ONNX
img = torch.zeros(1, 3, 320, 320)
torch.onnx.export(
    model,
    img,
    "best.onnx",
    opset_version=12,
    simplify=True,
    input_names=["images"],
    output_names=["output"],

    # 👇 这一行是关键！你之前漏了！导致输出格式完全错误！
    do_constant_folding=True,
)

print("✅ 正确导出 ONNX！")