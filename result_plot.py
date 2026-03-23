import pandas as pd
import matplotlib.pyplot as plt

# 你的结果路径
path = "runs/train/exp23/results.csv"

# 读取数据
df = pd.read_csv(path)

# 清理列名（去掉多余空格）
df.columns = [col.strip() for col in df.columns]

# 开始画图
plt.figure(figsize=(18, 12))

# 1 Box Loss
plt.subplot(2, 3, 1)
plt.plot(df['epoch'], df['train/box_loss'], label='train box loss')
plt.plot(df['epoch'], df['val/box_loss'], label='val box loss')
plt.title('Box Loss')
plt.legend()

# 2 Obj Loss
plt.subplot(2, 3, 2)
plt.plot(df['epoch'], df['train/obj_loss'], label='train obj loss')
plt.plot(df['epoch'], df['val/obj_loss'], label='val obj loss')
plt.title('Obj Loss')
plt.legend()

# 3 Cls Loss
plt.subplot(2, 3, 3)
plt.plot(df['epoch'], df['train/cls_loss'], label='train cls loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='val cls loss')
plt.title('Cls Loss')
plt.legend()

# 4 Precision + Recall
plt.subplot(2, 3, 4)
plt.plot(df['epoch'], df['metrics/precision'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall'], label='Recall')
plt.title('Precision / Recall')
plt.legend()

# 5 mAP (这里用你真实的列名！)
plt.subplot(2, 3, 5)
plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP 0.5')
plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP 0.5:0.95')
plt.title('mAP')
plt.legend()

plt.tight_layout()
plt.savefig("MY_FINAL_RESULT.png", dpi=300)
print("✅ 训练曲线图生成成功！")
print("📸 图片名：MY_FINAL_RESULT.png")