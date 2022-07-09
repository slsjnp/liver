import PIL.Image as Image
import os
import numpy as np
import nibabel as nib


def bmp_to_nii(bmp_path, nii_path, nii_name):
    filenames = os.listdir(bmp_path)
    seg_vol = np.zeros((len(filenames), 512, 512))
    for i, name in enumerate(filenames):
        name = os.path.join(bmp_path, name)
        img = Image.open(name)
        img = np.asarray(img)
        img_c = img.copy()
        img_c[img_c != 0] = 1
        seg_vol[i] = img_c
    seg_vol = np.asarray(seg_vol)
    seg_vol = np.transpose(seg_vol, [1, 2, 0])
    seg_vol = nib.Nifti1Image(seg_vol, affine=np.eye(4))
    nii_path = os.path.join(nii_path, nii_name)
    nib.save(seg_vol, nii_path)

# if __name__ == '__main__':
#     rip_path = "RIP/"
#     nii_path = "nii/"
#     bmp_to_nii(rip_path,nii_path)
