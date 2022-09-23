import numpy as np
import skimage
from .roi import one_roi

def skeletonize(img,spacing=None,surface=False):

    if spacing is None:
        spacing=[1,1,1]
    orig_img=img
    trimed_idx=one_roi(img,margin=2,return_index=True)
    img=img[trimed_idx]
    skel=np.zeros(orig_img.shape)
    spacing=np.array(spacing)
    spacing=spacing/spacing.min()
    
    img2s=skimage.transform.rescale(img,spacing,preserve_range=True,mode='edge')>0
    if surface:
        skel2=skimage.morphology.medial_surface(img2s)>0
    else:
        skel2=skimage.morphology.skeletonize_3d(img2s)>0
    skel[trimed_idx]=skimage.transform.resize(skel2,img.shape,preserve_range=True,mode='edge')>0
    
    return skel
