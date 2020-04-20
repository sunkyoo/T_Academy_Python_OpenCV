import sys
import cv2


filenames = ['namecard1.jpg', 'namecard2.jpg', 'namecard3.jpg']

for filename in filenames:
    src = cv2.imread(filename, cv2.IMREAD_COLOR)

    if src is None:
        print('Image load failed!')
        sys.exit()

    src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    th, dst = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('threshold:', th)

    cv2.imshow('src', src)
    cv2.imshow('src_gray', src_gray)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()
