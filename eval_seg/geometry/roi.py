import numpy as np
def one_roi(img,margin=0,*,ignore=[],threshold=10,return_index=False):
    allzeros = np.where((arr < threshold) &(arr!=0))
    idx = ()
    for i in range(len(allzeros)):
        if i in ignore:
            idx += (np.s_[:],)
        else:
            idx += (np.s_[max(0,allzeros[i].min() - margin):min(img.shape[i],allzeros[i].max() + margin + 1)],)
    if return_index:
        return idx
    return arr[idx]