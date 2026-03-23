import onnxruntime as ort
import cv2
import numpy as np
import os
from datetime import datetime

# ------------------------
# 配置
# ------------------------
onnx_path = "runs/train/exp23/weights/best.onnx"  # ONNX 模型路径
img_size = 320                                    # 输入模型尺寸
conf_thres = 0.4                                 # 置信度阈值
iou_thres = 0.45                                  # NMS 阈值
save_img_folder = "/home/spring/hand/yolov5/runs/test"                  # 保存图片的文件夹
save_label_folder = "/home/spring/hand/yolov5/runs/test"                # 保存标签文件的文件夹

# 创建文件夹（如果不存在）
os.makedirs(save_img_folder, exist_ok=True)
os.makedirs(save_label_folder, exist_ok=True)

# ------------------------
# 加载 ONNX 模型
# ------------------------
ort_session = ort.InferenceSession(onnx_path)

# ------------------------
# 类别名称列表（根据训练时的类别来设置）
# ------------------------
class_names = ['hand', 'other_class', 'another_class']  # 这里替换成你训练时的类别名称

# ------------------------
# 预处理函数
# ------------------------
def preprocess(img, img_size):
    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad)
    img_padded = cv2.copyMakeBorder(img_resized, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_padded = img_padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img_padded = np.ascontiguousarray(img_padded).astype(np.float32) / 255.0
    return img_padded[np.newaxis, :], r, dw, dh, w0, h0

# ------------------------
# NMS & 辅助函数
# ------------------------
def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2]/2
    y[:, 1] = x[:, 1] - x[:, 3]/2
    y[:, 2] = x[:, 0] + x[:, 2]/2
    y[:, 3] = x[:, 1] + x[:, 3]/2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    classes = prediction[:, 5]
    mask = scores > conf_thres
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
    boxes = xywh2xyxy(boxes)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], scores[indices], classes[indices]
    return np.array([]), np.array([]), np.array([])

# ------------------------
# 摄像头实时检测
# ------------------------
cap = cv2.VideoCapture(0)  # 打开摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头帧，退出...")
        break

    # 保存摄像头的原始帧
    original_frame = frame.copy()

    img_input, ratio, dw, dh, w0, h0 = preprocess(frame, img_size)

    # 推理
    outputs = ort_session.run(None, {"images": img_input})
    pred = outputs[0]  # (1, 6300, 6)

    # NMS
    boxes, scores, classes = non_max_suppression(pred[0], conf_thres, iou_thres)

    # 绘制检测框并显示类别名称
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        # 映射回原图
        x1 = max(0, int((x1 - dw) / ratio))
        y1 = max(0, int((y1 - dh) / ratio))
        x2 = min(w0, int((x2 - dw) / ratio))
        y2 = min(h0, int((y2 - dh) / ratio))

        # 获取类别名称
        class_name = class_names[int(cls)]  # 获取类别名称
        label = f'{class_name} {score:.2f}'  # 类别名称和置信度

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示实时检测结果
    cv2.imshow('Real-time Detection', frame)

    # 按 's' 键保存原始图像和标签文件
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # 获取当前时间作为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存原始图像（不带检测框）
        img_save_path = os.path.join(save_img_folder, f'{timestamp}.jpg')
        cv2.imwrite(img_save_path, frame)  # 保存原始图像
        
        # 保存 YOLO 格式的标签文件
        label_save_path = os.path.join(save_label_folder, f'{timestamp}.txt')
        with open(label_save_path, 'w') as f:
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                # 映射回原图
                x1 = max(0, (x1 - dw) / ratio)
                y1 = max(0, (y1 - dh) / ratio)
                x2 = min(w0, (x2 - dw) / ratio)
                y2 = min(h0, (y2 - dh) / ratio)
                # YOLO xywh 格式，归一化
                x_center = ((x1 + x2) / 2) / w0
                y_center = ((y1 + y2) / 2) / h0
                width = (x2 - x1) / w0
                height = (y2 - y1) / h0
                f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Image and label saved as {timestamp}.jpg and {timestamp}.txt")

    # 按 'q' 键退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()