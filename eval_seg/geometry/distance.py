import edt
import numpy as np
def distance(img,spacing=None,mode='in',mask=None):
    """
    mode=in,out,both
    """
    orig_img=img
    trimed_idx=np.s_[:,:,:]
    
    if not (mask is None):
        trimed_idx=one_roi(img,margin=2,return_index=True)
        img=img[trimed_idx]
    
    dst=np.zeros(orig_img.shape)
    if mode='both':
        dst[trimed_idx]=edt.sdf(img,anisotropy=spacing)
    elif mode=='in':
        trimed_idx=one_roi(img,margin=2,return_index=True)
        img=img[trimed_idx]
        dst[trimed_idx]=edt.edt(img,anisotropy=spacing)
    else:
        dst[trimed_idx]=edt.edt(~img,anisotropy=spacing)
    return dst
    
    