import cv2
import matplotlib.pyplot as plt
import numpy as np

# 定义一个图像显示函数
# name需要是一个字符串，img是一个图像变量
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # 1 读取图像
# img = cv2.imread("cat.jpg")
# print(img)
#