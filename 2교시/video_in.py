import sys
import cv2


# 동영상 파일로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture('vtest.avi')

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 프레임 해상도, 전체 프레임수, FPS 출력
print('Frame width:', round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

fps = round(cap.get(cv2.CAP_PROP_FPS))
print('FPS:', fps)

delay = round(1000 / fps)

# 매 프레임 처리 및 화면 출력
while True:
    ret, frame = cap.read()

    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150)

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(delay) == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
