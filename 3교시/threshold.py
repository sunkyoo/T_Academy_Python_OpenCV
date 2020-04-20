import sys
import cv2
import histogram


filename = 'namecard1.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]

src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('src', src)

hist_img = histogram.getGrayHistImage(histogram.calcGrayHist(src))
hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
cv2.imshow('hist_img', hist_img)


def on_threshold(pos):
    _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)
    hist_img2 = hist_img.copy()
    cv2.line(hist_img2, (pos, 0), (pos, 100), (0, 128, 255))
    cv2.imshow('hist_img', hist_img2)
    cv2.imshow('dst', dst)


cv2.namedWindow('dst')
cv2.createTrackbar('Threshold', 'dst', 0, 255, on_threshold)
cv2.setTrackbarPos('Threshold', 'dst', 130)

cv2.waitKey()
cv2.destroyAllWindows()
