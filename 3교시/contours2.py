import sys
import random
import numpy as np
import cv2

src = cv2.imread('namecard1.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

h, w = src.shape[:2]
dst1 = np.zeros((h, w, 3), np.uint8)
dst2 = np.zeros((h, w, 3), np.uint8)

# 이진화
_, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 외곽선 검출
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    pts = contours[i]

    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst1, contours, i, c, 1)

    # 너무 작은 객체는 제외
    if (cv2.contourArea(pts) < 1000):
        continue

    # 외곽선 근사화
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)

    # 컨벡스가 아니면 제외
    if not cv2.isContourConvex(approx):
        continue

    if len(approx) == 4:
        cv2.drawContours(dst2, contours, i, c, 2)


cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
