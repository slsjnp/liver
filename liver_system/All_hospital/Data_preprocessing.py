import os
import numpy
import pydicom
import torch
from All_hospital.WL import *
import PIL.Image as Image
import pydicom.uid
import re

def read_data(path):
    filenames = os.listdir(path)
    pattern = re.compile(r'\d+')
    filenames.sort(key=lambda filenames: int(pattern.findall(filenames)[-1]))
    slices = np.zeros((len(filenames), 512, 512))
    for i, name in enumerate(filenames):
        name = os.path.join(path, name)
        slice = pydicom.dcmread(name, force=True)
        slice.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        # a = slice.pixel_array
        slice = WL(slice, 150, 300)
        slices[i] = slice

    return slices, filenames



