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
# # 2 显示图像
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 3 定义一个图像显示函数
# def cv_show(name,img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# a = cv_show("image", img)

# # 4 查看图像的维数
# print(img.shape)

# # 5 读取图像的灰度图:cv2.IMREAD_GRAYSCALE可以用0来代替
# # cv2.COLOR_BGR2GRAY只使用在颜色转换空间中
# img = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)
# print(img)
# print(img.shape)
# cv2.imshow("image2", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 6 保存图像:必须要加.\保存在当前目录下
# cv2.imwrite(".\mycat111.jpg", img)

# # 7 查看图像的类型
# print(type(img))
# # 查看图像的像素点个数
# print(img.size)
# # 查看图像的数据编码形式
# print(img.dtype)

# 8 读取视频:vc = cv2.VideoCapture(0):表示打开电脑自带的摄像头"test.mp4"
vc = cv2.VideoCapture("test.mp4")
# vc = cv2.VideoCapture(0)
# 检查是否打开正确
if vc.isOpened():
    open, frame = vc.read()
    print(open)
else:
    open = False

# 循环输出视频中的每一帧图像
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        # cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
        # cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
        # cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
        # 也可以用数字1来代替，进行彩色读取，要读取灰度图，必须要用下面的命令
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("result", gray)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()

# # 9 部分数据提取
# img = cv2.imread("cat.jpg")
# # [0:200, 0:200] 表示的是这个图像矩阵的维数
# cat = img[0:200, 0:200]
# cv2.imshow("img", img)
# cv2.imshow("cat", cat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 要想使用下面的横向显示，两个图片的维数必须一致才行
# dog = img[0:200, 0:200]
# res = np.hstack((dog,cat))
# cv_show("res", res)

# # 10 颜色通道提取
# img = cv2.imread("cat.jpg")
# b,g,r = cv2.split(img)
# print(r)
# print(r.shape)
# # 进行通道合并
# img = cv2.merge((b,g,r))
# print(img.shape)

# # 11 保存单一通道颜色的图像
# img = cv2.imread("cat.jpg")
# # 保存R
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,1] = 0
# cv2.imshow("img", img)
# cv2.imshow("R", cur_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 保存G
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,2] = 0
# cv_show("G", cur_img)

# # 保存B
# cur_img = img.copy()
# cur_img[:,:,1] = 0
# cur_img[:,:,2] = 0
# cv_show("B", cur_img)

# # 12 边界填充
# img = cv2.imread("cat.jpg")
# top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
# # borderType：定义要添加边框的类型，它可以是以下的一种：
# # cv2.BORDER_CONSTANT：添加的边界框像素值为常数（需要额外再给定一个参数）
# # cv2.BORDER_REFLECT：添加的边框像素将是边界元素的镜面反射，类似于gfedcb|abcdefgh|gfedcba
# # cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT：和上面类似，但是有一些细微的不同，类似于gfedcb|abcdefgh|gfedcba
# # cv2.BORDER_REPLICATE：使用最边界的像素值代替，类似于aaaaaa|abcdefgh|hhhhhhh
# # cv2.BORDER_WRAP：不知道怎么解释，直接看吧，cdefgh|abcdefgh|abcdefg
# # value：如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值。
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# constant =cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)
#
# # 绘制子图
# plt.subplot(231)
# plt.imshow(img, "gray")
# plt.title("ORIGINAL")
#
# plt.subplot(232), plt.imshow(replicate, "gray"), plt.title("REPLICATE")
#
# plt.show()


# # 13 数值计算
# img_cat = cv2.imread("cat.jpg")
# img_dog = cv2.imread("dog.jpg")
# # print(img_cat)
# # 增加图像的像素值，而不是图像的尺寸
# img_cat2 = img_cat + 10
# # print(img_cat2)
# # numpy中的加法-cv中图像中每个像素点的值最大为255，大于它的话，将会取余数
# print(img_cat + img_cat2)
# # cv中的加法：如果数大于255，将去最大值255
# # print(cv2.add(img_cat,img_cat2))


# # 14 改变图像的尺寸
# img = cv2.imread("cat.jpg")
# img_cat = cv2.imread("cat.jpg")
# img_dog = cv2.imread("dog.jpg")
# print(img_cat.shape)
# print(img_dog.shape)
# # # # 重新改变狗的尺寸:这跟前面改变像素不一样，这可以改变图像的大小
# # 还有在使用cv2.resize时，行和列的顺序是相反的：(500, 414)，第一个数表示列，第二个数表示行
# img_dog1 = cv2.resize(img_dog, (500, 414))
# print(img_dog1.shape)
# cv_show("img_dog", img_dog)
# cv_show("img_dog1", img_dog1)
#
# # 进行图像的缩放
# res = cv2.resize(img, (0,0), fx=3, fy=1)
# plt.imshow(res)
# plt.show()
#
# # 进行图像的融合:注意：在图像融合时，两个图象尺寸必须要相同才行
# res = cv2.addWeighted(img_cat, 0.4, img_dog1, 0.6, 0)
# # plt.imshow(res)
# cv_show("rong", res)