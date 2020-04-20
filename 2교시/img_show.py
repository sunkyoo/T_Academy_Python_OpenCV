import sys
import cv2

# 영상 불러오기
img = cv2.imread('cat.bmp')

if img is None:
    print('Image load failed!')
    sys.exit()

# 영상 화면 출력
cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()

# 창 닫기
cv2.destroyAllWindows()
