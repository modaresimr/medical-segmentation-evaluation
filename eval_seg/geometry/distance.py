import edt
import numpy as np
from . import one_roi


def distance(img, spacing=None, mode='in', mask=None, ignore_distance_rate=1):
    """
    mode=in,out,both
    """
    spacing = spacing if not (spacing is None) else [1, 1, 1]
    orig_img = img

    if not (mask is None):
        trimed_idx = one_roi(mask, margin=4, return_index=True)
    else:
        trimed_idx = one_roi(img, margin=(np.array(img.shape) * ignore_distance_rate + 4).astype(int), return_index=True)

        # trimed_idx = np.s_[:, :, :]

    img = img[trimed_idx]

    dst = np.zeros(orig_img.shape, np.float16)
    if mode == 'both':
        dst[trimed_idx] = edt.edt(~img, anisotropy=spacing, black_border=True)

        trimed_idx = one_roi(img, margin=4, return_index=True)
        img = img[trimed_idx]
        dst[trimed_idx] -= edt.edt(img, anisotropy=spacing, black_border=True)
    elif mode == 'in':
        trimed_idx = one_roi(img, margin=4, return_index=True)
        img = img[trimed_idx]
        dst[trimed_idx] = edt.edt(img, anisotropy=spacing, black_border=True)
    else:
        dst[trimed_idx] = edt.edt(~img, anisotropy=spacing, black_border=True)

    return dst
