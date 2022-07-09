import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import skimage.io as sio
from vol_3d.recon_toolkit import recon_skimage


# #read a nii file from LITS dataset


def obj_make(nii_path, obj_path, name):
    nii_data = nib.load(nii_path)
    volume = nii_data.get_fdata()
    volume[volume > 1] = 1

    # #read a raw file from SLIVER07 dataset
    # volume = sio.imread('liver-seg001.mhd', plugin='simpleitk')

    recon_skimage(volume, obj_path, name, spacing=(1, 1, 1))
