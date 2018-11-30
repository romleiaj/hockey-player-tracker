import numpy as np
import cv2
import sys

name = str(sys.argv[1])
cap = cv2.VideoCapture(name)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    if ret == False:
        break

    fgmask = fgbg.apply(frame)
    edges = cv2.Canny(fgmask, 3500, 4500, apertureSize=5)

    cv2.imshow('frame',edges)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
