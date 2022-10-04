import k3d
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from .. import geometry


def ortho_slicer(img, pred, cut, args={}):
    row = len(pred)
    col = 3
    fig, axes = plt.subplots(row, col, figsize=(col * 2, row * 2), dpi=100)
    if row == 1:
        axes = [axes]

    if len(pred) == 0:
        pred['data'] = np.zeros(img.shape)

    for pi, p in enumerate(pred):
        for i in range(3):
            predcut, _ = geometry.slice(pred[p], None, i, cut[i])
            axes[pi][i].imshow(predcut, cmap='bone')

            if predcut.sum() > 0:
                axes[pi][i].contour(predcut, cmap='bone')
            if i == 0:
                axes[pi][i].axhline(cut[2])
                axes[pi][i].axvline(cut[1])
            if i == 1:
                axes[pi][i].axhline(cut[2])
                axes[pi][i].axvline(cut[0])
            if i == 2:
                axes[pi][i].axhline(cut[1])
                axes[pi][i].axvline(cut[0])
            # axes[i].invert_xaxis()
            axes[pi][i].set_ylim(0, predcut.shape[0])
            axes[pi][i].set_xlim(0, predcut.shape[1])
            # axes[i].invert_yaxis()
        axes[pi][1].set_title(f'{p} {cut}')
    if args.get('dst', ''):
        fig.savefig(args['dst'] + ".png")
    if args.get('show', 1):
        fig.show()
    else:
        plt.close()