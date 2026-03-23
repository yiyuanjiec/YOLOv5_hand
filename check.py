import os
import glob

# 自动匹配你的路径
train_img = "data/images/hand_detection_dataset/to_coco/train2017/"
val_img = "data/images/hand_detection_dataset/to_coco/val2017/"

def check_folder(img_dir):
    print(f"\n正在检查: {img_dir}")
    error_files = []
    txt_files = glob.glob(img_dir + "*.txt")

    for txt in txt_files:
        img = txt.replace(".txt", ".jpg")
        if not os.path.exists(img):
            error_files.append(f"【图片缺失】{txt}")
            continue

        with open(txt, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            try:
                # YOLO 标签格式：class x y w h
                cls, x, y, w, h = map(float, line.split())
                # 检查错误
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    error_files.append(f"【坐标负数/非法】{txt}")
                if x + w/2 > 1 or y + h/2 > 1 or x - w/2 < 0 or y - h/2 < 0:
                    error_files.append(f"【坐标越界】{txt}")
            except:
                error_files.append(f"【格式错误】{txt}")

    # 输出结果
    if not error_files:
        print("✅ 无错误！")
    else:
        print(f"❌ 共发现 {len(error_files)} 个错误：")
        for e in list(set(error_files))[:50]:  # 去重+显示前50
            print(e)

# 开始检查
check_folder(train_img)
check_folder(val_img)