
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data,filters
import cv2
import numpy as np

def sobel_grad(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)   # 转回uint8
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    ######img =color.rgb2gray(data.camera())
    ######denoised = filters.rank.median(img, morphology.disk(2)) #过滤噪声
    print(sobelxy)
    #将梯度值低于10的作为开始标记点
    #markers = filters.rank.gradient(sobelxy, morphology.disk(5)) <10
    #markers = ndi.label(markers)[0]
    #plt.imshow(markers)
    #plt.show()
    #gradient = filters.rank.gradient(sobelxy, morphology.disk(2)) #计算梯度
    #image, contours, hierarchy = cv2.findContours(sobelxy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    # labels =morphology.watershed(sobelxy, 2, mask=sobelxy) #基于梯度的分水岭算法
    '''
    watershed( InputArray image, InputOutputArray markers )
    第一个参数 image，必须是一个8bit 3通道彩色图像矩阵序列
    关键是第二个参数markers，在执行分水岭函数watershed之前，
    必须对第二个参数markers进行处理，它应该包含不同区域的轮廓，每个轮廓有一个自己唯一的编号。

    算法会根据markers传入的轮廓作为种子（也就是所谓的注水点），
    对图像上其他的像素点根据分水岭算法规则进行判断，并对每个像素点的区域归属进行划定，
    直到处理完图像上所有像素点。而区域与区域之间的分界处的值被置为“-1”，以做区分。
    '''
    labels1 = cv2.convertScaleAbs(sobelxy)
    ret, binary = cv2.threshold(labels1, 0, 255, cv2.THRESH_BINARY)
    #labels=markers==0
    return binary
#img = cv2.imread('./130_0544.bmp',0)

#labels=sobel_grad(img)
#gray = cv2.cvtColor(labels,cv2.COLOR_BGR2GRAY)


#print(ret)
#ret, binary = cv2.threshold(labels,0,255,cv2.THRESH_BINARY)
#image1, contours1, hierarchy1 = cv2.findContours(labels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.imshow('img',binary)
#cv2.waitKey(0)
#plt.imshow(binary)
#plt.show()
#创建窗口
'''
fig, axes = plt.subplots(2,2,figsize=(8, 8))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(img,plt.cm.gray)
ax0.set_title("Original")
ax1.imshow(sobelxy,plt.cm.nipy_spectral)
ax1.set_title("Gradient")
ax2.imshow(markers,plt.cm.nipy_spectral)
ax2.set_title("Markers")
ax3.imshow(labels,plt.cm.nipy_spectral)
ax3.set_title("Segmented")
#plt.cm.nipy_spectral实现的功能是给不同的类别不同的颜色

plt.figure()
plt.imshow(labels)
plt.show()
'''
'''
for ax in axes:
    ax.axis('off')

fig.tight_layout() #自动布局
plt.show()
'''
'''
分水岭在地理学上就是指一个山脊，水通常会沿着山脊的两边流向不同的“汇水盆”。
分水岭算法是一种用于图像分割的经典算法，是基于拓扑理论的数学形态学的分割方法。
如果图像中的目标物体是连在一起的，则分割起来会更困难，
分水岭算法经常用于处理这类问题，通常会取得比较好的效果。

分水岭算法也可以和梯度相结合，来实现图像分割。
一般梯度图像在边缘处有较高的像素值，而在其它地方则有较低的像素值，
理想情况下，分山岭恰好在边缘。因此，我们可以根据梯度来寻找分山岭

分水岭算法常用的操作步骤：
1.彩色图像灰度化
2.求梯度图
3.最后在梯度图的基础上进行分水岭算法，求得分段图像的边缘线
4.绘制分割出来的区域，可以使用随机颜色填充  
'''
