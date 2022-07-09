import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math
import cv2 as cv


def slope(x, y, count):
    x2 = x
    y2 = y
    x3 = []
    y3 = []
    θ = []
    tgθ3 = []
    # print(x2)
    # print(y2)
    for i in range(1, len(x2), count):
        # 这句容易导致数组越界 斟酌************************************************************
        if (i + count + 1 > len(x2) - 1):
            return np.var(θ) * 10
        a = b = 0
        # print('点数为：', i)
        t1 = x2[i + 1] - x2[i]
        t2 = x2[i + count + 1] - x2[i + count]

        x3.append(x2[i])
        y3.append(y2[i])

        while (1):
            # t3=y2[i+1+a]-y2[i]
            # t4=y2[i+count+1+b]-y2[i+count]
            if t1 == 0:
                a = a + 1
                t1 = x2[i + 1 + a] - x2[i]
            if t2 == 0:
                b = b + 1
                if i + count + 1 + b >= len(x2):
                    break
                t2 = x2[i + count + 1 + b] - x2[i + count]

            elif t1 != 0 and t2 != 0:
                break

        if i + count + 1 + b >= len(x2):
            break
        if (t1 == 0):
            continue
        if (t2 == 0):
            continue
        # print('(a,b)=:', a, b)
        #
        # print(x2[i], y2[i])
        # print(x2[i + 1 + a], y2[i + 1 + a])
        # print(x2[i + count], y2[i + count])
        # print(x2[i + count + 1 + b], y2[i + count + 1 + b])
        plt.scatter(x3, y3)
        # plt.show()
        # print(x2[i+1],y2[i+1])
        # k1=(y2[i+1]-y2[i])/(x2[i+1]-x2[i])
        # k2=(y2[i+count+1]-y2[i+count])/(x2[i+count+1]-x2[i+count])
        # if (x2[i+count+1+b]-x2[i+count])<0:

        k1 = (y2[i + 1 + a] - y2[i]) / (x2[i + 1 + a] - x2[i])
        k2 = (y2[i + count + 1 + b] - y2[i + count]) / (x2[i + count + 1 + b] - x2[i + count])
        # print(k1, k2)
        # if k1*k2+1==0:
        # continue
        if k1 <= 0 and k2 <= 0 or k1 > 0 and k2 > 0:
            tgθ = (abs(k2 - k1)) / (1 + k1 * k2)
        # tgθ1=atan(tgθ)
        elif k1 > 0 and k2 > (-1 / k1):
            tgθ = abs((k2 - k1) / (1 + k1 * k2))
        elif k1 > 0 and k2 < (-1 / k1):
            tgθ = (k2 - k1) / (1 + k1 * k2)
            tgθ = -tgθ
            # print(tgθ)
        elif k1 * k2 != -1:
            tgθ = (k2 - k1) / (1 + k1 * k2)

        # tgθ1=atan(tgθ)
        if k1 * k2 == -1:
            # tgθ=np.pi/2
            tgθ1 = 90
        else:
            tgθ1 = (atan(tgθ) * 180) / np.pi
        if tgθ1 < 0:
            tgθ1 = tgθ1 + 180
            # tgθ1=abs(tgθ1)
        # print('夹角为：')
        # if x2[i+count+1+b]< x2[i+count] and x2[i+1+a] > x2[i] and k1*k2 <0 or  x2[i+count+1+b]> x2[i+count] and x2[i+1+a] < x2[i] and k1*k2 <0:
        # tgθ1=180-tgθ1
        # print('夹角为：', tgθ1)
        tgθ3.append(tgθ1)
        tgθ2 = (tgθ1 / 180)  # 归一化
        # print(tgθ1)
        θ.append(tgθ2)
        # print(i)
        # print(i + count)
        # print(len(x2))

    # print('角度为：', tgθ3)
    return np.var(θ) * 10


'''
x = [40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135]
y = [536,529,522,516,511,506,502,498,494,490,487,484,481,478,475,472,470,467,465,463]
f=slope(x,y,1)
print('平滑度为：',f)
'''
'''
def slope(x,y,count1,count2):
    x2=x
    y2=y
    θ=[]
    tgθ3=[]
    #print(x2)

    #print(y2)
    for i in range(1,len(x),count1):

        if (i>=len(x2)-(i+count1+count2)):
            break
        a=b=0
        print('点数为：',i)
        if (x2[i+count2]-x2[i]==0):
            continue
        if (x2[i+count1+count2]-x2[i+count1]==0):
            continue
        print('(a,b)=:',a,b)
        print(x2[i],y2[i])
        print(x2[i+count2],y2[i+count2])
        print(x2[i+count1],y2[i+count1])
        print(x2[i+count1+count2],y2[i+count2+count2])
        #print(x2[i+1],y2[i+1])
        #k1=(y2[i+1]-y2[i])/(x2[i+1]-x2[i])
        #k2=(y2[i+count+1]-y2[i+count])/(x2[i+count+1]-x2[i+count])
        k1=(y2[i+count2]-y2[i])/(x2[i+count2]-x2[i])
        k2=(y2[i+count1+count2]-y2[i+count1])/(x2[i+count1+count2]-x2[i+count1])
        print(k1,k2)
        if k1*k2+1==0:
            continue
        if k1 <=0 and k2 <=0 or k1 >0 and k2>0:
            tgθ=(abs(k2-k1))/(1+k1*k2)
        #tgθ1=atan(tgθ)
        elif k1>0 and k2 > (-1/k1):
            tgθ=abs((k2-k1)/(1+k1*k2))
        elif k1>0 and k2 < (-1/k1):
            tgθ=(k2-k1)/(1+k1*k2)
            tgθ=-tgθ
            #print(tgθ)
        else :
            tgθ=(k2-k1)/(1+k1*k2)
        #tgθ1=atan(tgθ)
        tgθ1=(atan(tgθ)*180)/np.pi
        if  tgθ1<0:
            tgθ1=tgθ1+180
            #tgθ1=abs(tgθ1)
        #print('夹角为：')
        print(tgθ1)
        tgθ3.append(tgθ1)
        tgθ2=(tgθ1/180) #归一化
        #print(tgθ1)
        θ.append(tgθ2)
    print('角度为：',tgθ3)
    return np.var(θ)*10
    '''


def rotate(img):
    '''
    在一定角度范围内，图像随机旋转
    :param img:
    :param limit_up:旋转角度上限
    :param limit_down: 旋转角度下限
    :return: 旋转后的图像
    '''
    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    M = cv.getRotationMatrix2D(center_coordinate, 90, 1)

    # 仿射变换
    out_size = (cols, rows)
    rotate_img = cv.warpAffine(img, M, out_size, borderMode=cv.BORDER_REPLICATE)
    return rotate_img
    '''
img=cv.imread('./141_5.png')
img1=np.rot90(img)
cv.imwrite('./145_ro.png', img1)
'''