# https://ultralytics.com/images/bus.jpg 中の物体を抽出し，その領域を元の画像に描画せよ
import cv2
from ultralytics import YOLO


model = YOLO("yolov8x.pt")


results = model.predict(r"C:\Users\kotar\Documents\領域実習b\ex1-26.png", conf=0.225)


# 入力画像
img = results[0].orig_img


# 認識した物体領域を取得する．
boxes = results[0].boxes

# 人物の数を数えるための変数
person_count = 0


for box in boxes:
    # 物体領域のxy座標を取得する．
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # 物体のクラスIDを取得する．
    class_id = int(box.cls[0])

    # クラスIDが0なら人物
    if class_id == 0:
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