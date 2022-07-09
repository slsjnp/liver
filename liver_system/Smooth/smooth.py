import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import morphology
from .yt import slope

from .watershed1 import sobel_grad

'''
x1 = 0
xf = 512
y1 = 400
yf = 512
'''
pathout = './data_RIP/test1/'
# pathin='./data_RIP/img/070.png'
# pathin_mask='./data_RIP/mask/070.bmp'
pathin = '17.png'
pathin_mask = '17.bmp'

name = pathin[-7:-4]
print(name)
# 当鼠标按下时为True
drawing = False
# 如果mode为true时绘制矩形，按下'm'变成绘制曲线
mode = True
ix, iy = -1, -1
xf, yf = 0, 0


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, xf, yf, x1, y1
    # 当按下左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当左键按下并移动时绘制图形，event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        xf, yf = x, y
        # if drawing == True:
        # mode = True
        # cv2.rectangle(img,(ix,iy),(xf,yf),(0,255,0),1)   #第一个参数：img是原图  第二个参数：（x，y）是矩阵的左上点坐标 第三个参数：（x+w，y+h）是矩阵的右下点坐标 第四个参数：（0,255,0）是画线对应的rgb颜色第五个参数：2是所画的线的宽度
        # else:
        # 绘制圆圈，小圆点连在一起就成了线，3代表笔画的粗细
        # cv2.circle(img,(x,y),3,(0,0,255),-1)

    # 当鼠标松开时停止绘图
    elif event == cv2.EVENT_LBUTTONUP:
        xf, yf = x, y
        drawing = True
        cv2.rectangle(imgc, (ix, iy), (xf, yf), (0, 255, 0), 3)
        ###cv2.imwrite(pathout+name+'.bmp',imgc)
        print(ix)
        print(iy)
        print(xf)
        print(yf)
        return ix, iy, xf, yf


# cv2.dilate
'''
#下面把回调函数与OpenCV窗口绑定在一起，在主循环中奖'm'键与模式转换绑定在一起
'''


def check(img):
    # img = cv2.imread('0_0058.bmp')
    # cv2.imshow('image',img)
    img1 = img
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', draw_circle)

    while (1):
        cv2.imshow('img', img)
        k = cv2.waitKey(1)
        if k == ord('m'):
            mode = not mode
        elif k == ord('\r'):
            break
    return ix, iy, xf, yf


# cv2.destroyAllWindows()

# plt.figure(1)
# img = cv2.imread('E:/Download/data/slices/126.png',0)


def smooth(img, img_mask, x1, y1, xf, yf):
    a = img_mask
    img1_mask_1 = sobel_grad(img_mask)
    img1 = img[y1:yf, x1:xf]  # 在原图范围内裁剪
    img1_mask_s = img1_mask_1[y1:yf, x1:xf]

    # cv2.imshow('1_1',img1)
    # cv2.imshow('1_2',img1_mask_s)

    # cv2.waitKey(0)
    # img1_mask_s=sobel_grad(img1_mask)
    kernel = np.ones((2, 2), np.uint8)
    img1_di = cv2.dilate(img1_mask_s, kernel, iterations=1)

    img_x = img1 * img1_di

    ret, img_x = cv2.threshold(img_x, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow('2_1', img_x)
    if abs(xf - x1) > abs(yf - y1):
        img_ro = np.rot90(img_x)
    else:
        img_ro = img_x
    img_ro[img_ro == 255] = 1
    edge_s = morphology.skeletonize(img_ro)
    edge_s = edge_s.astype(np.uint8) * 255
    a = np.argwhere(edge_s>1)
    # edge_s = sobel_grad(img_ro)
    # cv2.imshow('2_2',edge_s)
    # edge_s = sobel_grad(edge_s)

    # cv2.waitKey(0)

    # edge_s[edge_s==255] = 1
    # skeleton01 = morphology.skeletonize(edge_s)
    # skeleton1 = skeleton01.astype(np.uint8)*255
    contours, hierarchy = cv2.findContours(edge_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>1:
        contours = contours[1]
    Lr = np.squeeze(contours)
    # max1 = np.max(Lr[:, 1])
    # m = np.argwhere(Lr[:, 1] == max1)

    # n = m[0][0]
    # print(m,n)
    # L1 = Lr[:n, :n]

    n1 = int(len(Lr) / 30)
    f = slope(Lr[:, 0], Lr[:, 1], n1)
    return f


if __name__ == "__main__":
    img = cv2.imread(pathin, 0)
    imgc = cv2.imread(pathin, 1)
    img_mask = cv2.imread(pathin_mask, 0)
    # print(img_mask)
    # img = cv2.imread('./pho/00065.bmp',0)
    # img = cv2.imread('1.png',0)
    cv2.imshow('original_img', img)

    x1, y1, xf, yf = check(imgc)
    # x1 = 40
    # y1 = 57
    # xf = 240
    # yf = 207
    s = smooth(img, img_mask, x1, y1, xf, yf)
    # print("平滑度为：", s)