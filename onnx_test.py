import onnxruntime as ort
import cv2
import numpy as np

# ------------------------
# 配置
# ------------------------
onnx_path = "runs/train/exp23/weights/best.onnx"  # ONNX 模型路径
img_path = "/home/spring/hand/yolov5/data/test/000000002658.jpg"                             # 测试图片路径
img_size = 320                                    # 输入模型尺寸
conf_thres = 0.4                                 # 置信度阈值
iou_thres = 0.45                                  # NMS 阈值

# ------------------------
# 加载 ONNX 模型
# ------------------------
ort_session = ort.InferenceSession(onnx_path)

# ------------------------
# 读取图片 & 预处理
# ------------------------
def preprocess(img_path, img_size):
    img = cv2.imread(img_path)
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
    return img_padded[np.newaxis, :], img, r, dw, dh

img_input, img_orig, ratio, dw, dh = preprocess(img_path, img_size)

# ------------------------
# 推理
# ------------------------
outputs = ort_session.run(None, {"images": img_input})
pred = outputs[0]  # (1, 6300, 6)

# ------------------------
# NMS
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
    # 将 xywh 转为 xyxy
    boxes = xywh2xyxy(boxes)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], scores[indices], classes[indices]
    return np.array([]), np.array([]), np.array([])

boxes, scores, classes = non_max_suppression(pred[0], conf_thres, iou_thres)

# ------------------------
# 绘制检测框
# ------------------------
for box, score, cls in zip(boxes, scores, classes):
    x1, y1, x2, y2 = box
    # 将坐标映射回原图
    x1 = max(0, int((x1 - dw) / ratio))
    y1 = max(0, int((y1 - dh) / ratio))
    x2 = min(img_orig.shape[1], int((x2 - dw) / ratio))
    y2 = min(img_orig.shape[0], int((y2 - dh) / ratio))
    cv2.rectangle(img_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_orig, f'{int(cls)} {score:.2f}', (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ------------------------
# 显示 & 保存
# ------------------------
cv2.imshow("ONNX Detection", img_orig)
cv2.imwrite("result.jpg", img_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()