import cv2
from pathlib import Path
from ultralytics import YOLO

# bus.jpg を同じフォルダに置いてある場合
image_path = Path(__file__).parent / "ex2-26.png"

model = YOLO("yolo11m.pt")

# 全クラスを検出する
results = model.predict(
    str(image_path),
    conf=0.28,
    iou=0.4,      # 0.75 より通常は 0.45〜0.6 が扱いやすい
    imgsz=1504,
    verbose=False,
)

img = results[0].orig_img
boxes = results[0].boxes
names = results[0].names

img_height, img_width = img.shape[:2]
object_count = 0

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    class_name = names[class_id]
    confidence = float(box.conf[0])

    # 極端に小さい検出だけ除外（必要なければ削除）
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < img_width * 0.01 or box_height < img_height * 0.01:
        continue

    object_count += 1

    # 枠を描画
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        thickness=3,
    )

# 検出数を表示
cv2.putText(
    img,
    f"Objects: {object_count}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2,
    cv2.LINE_AA,
)

print(f"検出した物体数: {object_count}")

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()