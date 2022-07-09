import numpy as np 
import pydicom
from functools import reduce
import os

"""
输入：
data: h*w*d 的int型三维numpy数组，前景为1，背景为0
voxel_spacing: 体素的尺寸信息 [h, w, d]，从任意dcm中获得
输出：
v: 以毫升为单位的体积值
"""
def count_voxels(data, voxel_spacing):
    num = len(np.where(data==1)[0])
    v = num * reduce(lambda x, y: x * y, voxel_spacing)
    v = v / 1000.
    return v

# 从任意dcm中读取空间信息
filepath = os.getcwd()
filename = filepath + '/liver_system/domo/app/static/upload_dcm'
filenames = os.listdir(filename)
sum = 0
for i in filenames:
    print(i[-8:])
    dcm = pydicom.dcmread(filename + '/' +i)
    voxel_spacing = dcm.PixelSpacing
    voxel_spacing.append(dcm.SliceThickness)
    voxel_spacing = map(lambda x: float(x), voxel_spacing)

    dummy_input = np.random.randint(0, 2, [512 ,512, 64])

    # 使用count_voxels计算体积，
    v = count_voxels(dummy_input, voxel_spacing)
    sum += v
print(sum)