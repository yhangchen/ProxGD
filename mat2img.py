import csv
import cv2
import numpy as np

name = "img"
lst = [3, 2, 1]
A = np.zeros((512, 512, 3))
for i in range(3):
    data = csv.reader(open("image/" + name + str(lst[i]) + ".csv"))
    line = 0
    for each in data:
        if each[0][0] not in "1234567890":
            continue
        else:
            A[line, :, i] = list(round(float(each_item)) for each_item in each)
        line += 1

cv2.imwrite(name + "_rewrite.png", A, [int(cv2.IMWRITE_JPEG_QUALITY),100])
