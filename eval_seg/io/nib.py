import nibabel as nib
import numpy as np


def read_nib(f):
    try:
        nib_data = nib.load(f)
        data = nib_data.dataobj[...]
        voxelsize = np.array(nib.affines.voxel_sizes(nib_data.affine))
        return data, voxelsize
    except:
        print(f'reading {f} failed!')
        return None, None
