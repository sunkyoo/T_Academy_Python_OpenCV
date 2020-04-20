import sys
import numpy as np
import cv2


def calcGrayHist(img):
    channels = [0]
    histSize = [256]
    histRange = [0, 256]

    hist = cv2.calcHist([img], channels, None, histSize, histRange)

    return hist


def getGrayHistImage(hist):
    histMax = np.max(hist)

    imgHist = np.full((110, 256), 255, dtype=np.uint8)
    for x in range(256):
        pt1 = (x, 100)
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))
        cv2.line(imgHist, pt1, pt2, 0)
        cv2.line(imgHist, (x, 101), (x, 110), x)

    #cv2.rectangle(imgHist, (0, 100), (255, 111), 0)

    return imgHist


if __name__ == '__main__':
    src = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        sys.exit()

    cv2.imshow('src', src)

    hist = calcGrayHist(src)
    histImg = getGrayHistImage(hist)
    cv2.imshow('histImg', histImg)
    cv2.imshow('histImg', getGrayHistImage(calcGrayHist(src)))
    cv2.waitKey()

cv2.destroyAllWindows()
