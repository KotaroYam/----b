# https://ultralytics.com/images/bus.jpg 中の物体を抽出し，その領域を元の画像に描画せよ
import cv2
from pathlib import Path
from ultralytics import YOLO


# ex2.py と同じフォルダにある画像を指定する
image_path = Path(__file__).parent / "ex2-26.png"


model = YOLO("yolov8x.pt")


# 人物のみを検出する（classes=[0] は person）
results = model.predict(
    str(image_path),
    conf=0.45,       # 誤検出を減らす（0.4〜0.6で調整）
    iou=0.35,        # 重複検出を減らす
    imgsz=1280,      # 小さい人物も見逃しにくくする
    classes=[0],     # person のみ
    verbose=False,
)


# 入力画像
img = results[0].orig_img


# 認識した物体領域を取得する．
boxes = results[0].boxes

# クラスIDをクラス名に変換する辞書
names = results[0].names

# 画像の高さを取得する
img_height, img_width = img.shape[:2]

# 人物の数を数える
person_count = 0


for box in boxes:
    # 物体領域のxy座標を取得する．
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 物体のクラスIDを取得する．
    class_id = int(box.cls[0])

    # 検出枠の幅と高さを求める
    box_width = x2 - x1
    box_height = y2 - y1

    # 明らかに小さすぎる誤検出を除外する
    if box_width < img_width * 0.02 or box_height < img_height * 0.05:
        continue

    # クラス名で人物かどうかを判定する
    if names[class_id] == "person":
        person_count += 1

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            thickness=3,
        )


# 人物数を画像に表示する
cv2.putText(
    img,
    f"Person: {person_count}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2,
    cv2.LINE_AA,
)


print(f"人物の数: {person_count}")


cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()