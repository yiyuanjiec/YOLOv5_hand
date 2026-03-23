import json
import sys

if len(sys.argv) < 2:
    print("用法: python show_coco.py 图片名.jpg")
    sys.exit(1)

target = sys.argv[1]

# 你的JSON路径
json_path = "/home/spring/hand/yolov5/data/images/hand_detection_dataset/to_coco/annotations/instances_train2017.json"

data = json.load(open(json_path))

# 找图片
img_id = None
for img in data["images"]:
    if img["file_name"] == target:
        img_id = img["id"]
        print("✅ 图片:", target)
        print("🆔 ID:", img_id)
        print("📏 尺寸:", img["width"], "x", img["height"])
        break

if not img_id:
    print("❌ 找不到图片")
    sys.exit(1)

# 找标注
print("\n📍 标注:")
for ann in data["annotations"]:
    if ann["image_id"] == img_id:
        print(ann["bbox"])