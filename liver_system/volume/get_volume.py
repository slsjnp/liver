import numpy as np
import os
import pydicom
import imageio
from functools import reduce


def get_array(path):
    names = os.listdir(path)
    array = []
    for name in names:
        img = imageio.imread(os.path.join(path, name))
        array.append(img)
    array = np.stack(array, 2)
    array[array != 0] = 1
    return array


def get_voxel_spacing(path):
    names = os.listdir(path)
    dcm = pydicom.dcmread(os.path.join(path, names[0]))
    voxel_spacing = dcm.PixelSpacing
    voxel_spacing.append(dcm.SliceThickness)
    voxel_spacing = list(map(lambda x: float(x), voxel_spacing))
    return voxel_spacing


def volume_calculator(array, voxel_spacing):
    num = len(np.where(array == 1)[0])
    v = num * reduce(lambda x, y: x * y, voxel_spacing)
    v = v / 1000.
    return v


def liver_tumor_spleen(dcm_path, liver_path, liver_tumor_path, spleen_path):
    voxel_spacing = get_voxel_spacing(dcm_path)

    liver_array = get_array(liver_path)
    liver_tumor_array = get_array(liver_tumor_path)
    spleen_array = get_array(spleen_path)

    v_liver = volume_calculator(liver_array, voxel_spacing)
    v_liver_tumor = volume_calculator(liver_tumor_array, voxel_spacing)
    v_spleen = volume_calculator(spleen_array, voxel_spacing)

    return v_liver - v_liver_tumor, v_liver / (v_spleen + 0.0001), v_spleen, v_liver_tumor


# filepath = os.getcwd()
# dcm = filepath + '/liver_system/domo/app/static/upload_dcm'
# liver = filepath + '/liver_system/domo/app/static/liver/bmp_nii'
# tumor = filepath + '/liver_system/domo/app/static/tumor/bmp_nii'
# spleen = filepath + '/liver_system/domo/app/static/spleen/bmp_nii'
# volume, scale = liver_tumor_spleen(dcm, liver, tumor, spleen)
# print(volume, scale)
