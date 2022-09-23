import json
import numpy as np
import compress_pickle
precompute_dir='precompute'
def get_precompute(name):
    pcf = f'{precompute_dir}/{name}.pkl.lz4'
    if not os.path.isfile(pcf):
        return None
    try:
        return compress_pickle.load(pcf)
    except Error as e:
        print('precompute file is not readable',e)
        return None
        
        
    
def save_precompute(name,data):
    os.makedirs(f'{precompute_dir}', exist_ok=True)
    pcf = f'{precompute_dir}/{name}.pkl.lz4'
    compress_pickle.dump(data, pcf)

    

class CircleList(list):
    def __getitem__(self, item):
        if type(item)==int:
            return super().__getitem__(item % len(self))
        return super().__getitem__(item)
    
def array_trim(arr, ignore=[], margin=0,threshold=10,return_index=False):
    all = np.where((arr < threshold) &(arr!=0))
    idx = ()
    for i in range(len(all)):
        if i in ignore:
            idx += (np.s_[:],)
        else:
            idx += (np.s_[all[i].min() - margin:all[i].max() + margin + 1],)
    if return_index:return idx
    return arr[idx]