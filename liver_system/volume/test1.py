from PIL import Image
import numpy as np
import pydicom
import os


def get_pixels_No(bmp_data_dir):
    pixels_No = 0
    bmp_files = os.listdir(bmp_data_dir)
    for bmp in bmp_files:
        bmp_file = os.path.join(bmp_data_dir, bmp)
        img = Image.open(bmp_file)
        img_array = np.array(img)
        # img_array.dtype为布尔类型，需要转换为Int类型，其累加和恰好为体素总和
        img_array_int = img_array.astype(int)
        pixels_No = pixels_No + img_array_int.sum()
    return pixels_No


def get_pixel_info(dcm_data_dir):
    pixel_infos = []
    dcm_files = os.listdir(dcm_data_dir)

    dcm_file_1 = os.path.join(dcm_data_dir, dcm_files[0])
    dcm_tag_1 = pydicom.read_file(dcm_file_1)
    # 获取像素间距.
    spacex, spacey = dcm_tag_1.PixelSpacing
    # 获取层间距
    # 有些 dcm图像并不是按照InstanceNumber进行排序的，不能直接用最后一张的slicelocation减去第一张，再除以张数
    SliceLocations = []
    ImagePositon_z = []
    for dcm in dcm_files:
        dcm_file = os.path.join(dcm_data_dir, dcm)
        dcm_tag = pydicom.read_file(dcm_file)
        SliceLocations.append(dcm_tag.SliceLocation)
        ImagePositon_z.append(dcm_tag.ImagePositionPatient[2])
    SliceLocations_max = max(SliceLocations)
    SliceLocations_min = min(SliceLocations)
    ImagePositon_z_max = max(ImagePositon_z)
    ImagePositon_z_min = min(ImagePositon_z)
    print(SliceLocations_max)
    print(SliceLocations_min)
    print(ImagePositon_z_max)
    print(ImagePositon_z_min)
    if SliceLocations_max - SliceLocations_min < 1e-10:
        spacez = abs(ImagePositon_z_max - ImagePositon_z_min) / (len(dcm_files) - 1)
    else:
        spacez = abs(SliceLocations_max - SliceLocations_min) / (len(dcm_files) - 1)
    pixel_infos = [spacex, spacey, spacez]

    return pixel_infos


def get_volume(dcm_data_dir, bmp_data_dir):
    pixel_infos = get_pixel_info(dcm_data_dir)
    pixels_No = get_pixels_No(bmp_data_dir)
    volume = pixel_infos[0] * pixel_infos[1] * pixel_infos[2] * pixels_No / 1000
    return volume


# dcm = pydicom.read_file(r"E:\20181210090945_LENG HONGYING F-44Y\Venous\0000.dcm")
# print(dcm)
# print(dcm.ImagePositionPatient[2])
# print(dcm[0x0020, 0x0032].keyword,dcm[0x0020, 0x0032].value)
filepath = os.getcwd()
filename = filepath + '/liver_system/domo/app/static/upload_dcm'
filenameb = filepath + '/liver_system/domo/app/static/result_bmp'
dcm_dir = filename
bmp_dir = filenameb
volume = get_volume(dcm_dir,
                    bmp_dir)
print("体积为%.1f" % volume)