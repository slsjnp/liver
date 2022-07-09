import numpy as np
import skimage.io as sio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def recon_skimage(volume, obj_path, name, spacing=(1., 1., 1.)):
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume, 0, spacing)
    faces += 1
    thefile = open(obj_path + '//' + name, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

    for item in faces:
        thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))

    thefile.close()
