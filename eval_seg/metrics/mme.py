from bz2 import compress
import skimage

from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion
import cc3d
import hashlib

import edt
from .. import geometry
import copy
import numpy as np
from . import MetricABS
from .. import common
from ..common import Cache
from sparse import COO

epsilon = 0.00001


class MME(MetricABS):

    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    def calculate_info(self, reference, spacing=None, num_classes=2, **kwargs):
        helper = {}

        helper['voxel_volume'] = spacing[0] * spacing[1] * spacing[2]
        helper['class'] = {}
        for c in range(num_classes):
            # print(f'class={c}')

            refc = reference == c

            gt_labels, gN = cc3d.connected_components(refc, return_N=True)
            gt_labels = gt_labels.astype(np.uint8)
            helperc = {}
            helper['class'][c] = helperc
            helperc['gt_labels'] = gt_labels
            helperc['gN'] = gN

            gt_regions = geometry.expand_labels(gt_labels, spacing=spacing).astype(np.uint8)

            helperc['components'] = {}
            for i in range(1, gN + 1):
                gt_component = gt_labels == i
                gt_region = gt_regions == i
                gt_border = geometry.find_binary_boundary(gt_component, mode='thick')
                gt_no_border = gt_component & ~gt_border
                gt_with_border = gt_component | gt_border
                in_dst = geometry.distance(gt_no_border, spacing=spacing, mode='in')
                out_dst = geometry.distance(gt_with_border, spacing=spacing, mode='out')
                gt_dst = out_dst + in_dst

                skeleton = geometry.skeletonize(gt_component, spacing=spacing) > 0
                skeleton_dst = geometry.distance(skeleton, spacing=spacing, mode='out')

                normalize_dst_inside = in_dst / (skeleton_dst + in_dst + epsilon)
                normalize_dst_outside = np.maximum(0, out_dst - epsilon / (skeleton_dst - out_dst + epsilon))
                # normalize_dst_outside = normalize_dst_outside.clip(0, normalize_dst_outside.max())
                normalize_dst = normalize_dst_inside + normalize_dst_outside

                helperc['components'][i] = {
                    'gt': gt_component,
                    'gt_region': gt_region,
                    'gt_border': gt_border,
                    'gt_dst': gt_dst,
                    'gt_out_dst': out_dst,
                    'gt_in_dst': in_dst,
                    'gt_skeleton': skeleton,
                    'gt_skeleton_dst': skeleton_dst,
                    'skgt_normalized_dst': normalize_dst,
                    'skgt_normalized_dst_in': normalize_dst_inside,
                    'skgt_normalized_dst_out': normalize_dst_outside
                }
                # if self.debug.get('show_precompute', 0):
                #     self.debug_helper(helperc['components'][i])

        return helper

    def evaluate(self, test, return_debug_data=False):
        reference = self.reference
        helper = self.helper
        assert test.shape == reference.shape, 'reference and test are not match'

        alpha1 = .1
        alpha2 = 1
        m_def = {d: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for d in ['D', 'B', 'U', 'R', 'T']}

        res = {'class': {}}

        data = {}

        for c in range(self.num_classes):
            data[c] = dc = common.Object()

            dc.testc = test == c
            dc.helperc = helper['class'][c]

            res['class'][c] = resc = {'total': {}, 'components': {}}

            dc.gt_labels, dc.gN = dc.helperc['gt_labels'], dc.helperc['gN']
            dc.pred_labels, dc.pN = cc3d.connected_components(dc.testc, return_N=True)

            resc['total'] = copy.deepcopy(m_def)
            dc.gts = {}
            for i in range(1, dc.gN + 1):
                dc.gts[i] = dci = common.Object()
                hci = dc.helperc['components'][i]
                dci.component_gt = hci['gt']
                dci.component_pred, dci.pred_comp = _get_component_of(dc.pred_labels, dc.pred_labels[dci.component_gt], dc.pN)
                dci.rel_gt_comps, dci.rel_gts = _get_component_of(dc.gt_labels, dc.gt_labels[dci.component_pred], dc.gN)

                dci.pred_in_region = dci.component_pred
                # if a prediction contains two gt only consider the part related to gt
                for l in dci.rel_gts:
                    if l != i:
                        dci.pred_in_region = dci.pred_in_region & ~dc.helperc['components'][l]['gt_region']

                dci.border_gt = hci['gt_border']
                dci.border_pred = geometry.find_binary_boundary(dci.pred_in_region, mode='inner')

                dci.dst_border_gt2pred = hci['gt_dst'][dci.border_pred]
                dci.dst_border_gt2pred_abs = np.abs(dci.dst_border_gt2pred)
                # dci.dst_border_gt2pred_v = np.zeros(dci.border_pred.shape)
                # dci.dst_border_gt2pred_v[dci.border_pred] = hci['gt_dst'][dci.border_pred]

                dci.gt_hd = dci.dst_border_gt2pred_abs.max() if len(dci.dst_border_gt2pred) > 0 else np.nan
                dci.gt_hd_avg = dci.dst_border_gt2pred_abs.mean() if len(dci.dst_border_gt2pred) > 0 else np.nan
                dci.gt_hd95 = np.quantile(dci.dst_border_gt2pred_abs, 0.95) if len(dci.dst_border_gt2pred) > 0 else np.nan

                dci.pred_border_dst = geometry.distance(dci.component_pred, mode='both', mask=dci.rel_gt_comps | dci.component_pred)

                dci.dst_border_pred2gt = dci.pred_border_dst[dci.border_gt]
                dci.dst_border_pred2gt_abs = np.abs(dci.dst_border_pred2gt)

                # dci.dst_border_pred2gt_v = np.zeros(dci.border_pred.shape)
                # dci.dst_border_pred2gt_v[dci.border_gt] = dci.pred_border_dst[dci.border_gt]

                dci.pred_hd = dci.dst_border_pred2gt_abs.max() if len(dci.dst_border_pred2gt) > 0 else np.nan

                dci.pred_hd_avg = dci.dst_border_pred2gt_abs.mean() if len(dci.dst_border_pred2gt) > 0 else np.nan

                dci.pred_hd95 = np.quantile(dci.dst_border_pred2gt_abs, 0.95) if len(dci.dst_border_pred2gt) > 0 else np.nan

                dci.hd = np.mean([dci.gt_hd, dci.pred_hd])
                dci.hd_avg = np.mean([dci.gt_hd_avg, dci.pred_hd_avg])
                dci.hd95 = np.mean([dci.gt_hd95, dci.pred_hd95])

                dci.skgtn_dst_in = hci['skgt_normalized_dst_in']
                dci.border_pred_inside_gt = dci.border_pred & dci.component_gt
                dci.border_pred_outside_gt = dci.border_pred & (~dci.component_gt)
                dci.skgtn_dst_pred_in = dci.skgtn_dst_in[dci.border_pred_inside_gt]
                # dci.skgtn_dst_pred_in = dci.skgtn_dst_pred_in[dci.skgtn_dst_pred_in > 0]

                if return_debug_data:
                    dci.skgtn_dst_pred_in_v = np.zeros(dci.skgtn_dst_in.shape)
                    dci.skgtn_dst_pred_in_v[dci.border_pred_inside_gt] = dci.skgtn_dst_in[dci.border_pred_inside_gt]

                dci.skgtn_dst_out = hci['skgt_normalized_dst_out']
                dci.skgtn_dst_pred_out = dci.skgtn_dst_out[dci.border_pred_outside_gt]

                if return_debug_data:
                    dci.skgtn_dst_pred_out_v = np.zeros(dci.skgtn_dst_out.shape)
                    dci.skgtn_dst_pred_out_v[dci.border_pred_outside_gt] = dci.skgtn_dst_out[dci.border_pred_outside_gt]

                dci.skgtn_dst_pred = np.concatenate([dci.skgtn_dst_pred_out, dci.skgtn_dst_pred_in])
                dci.skgtn_dst_pred = dci.skgtn_dst_pred[dci.skgtn_dst_pred > 0]

                dci.boundary_fp = min(1, dci.skgtn_dst_pred_out.mean()) if len(dci.skgtn_dst_pred_out) > 0 else 0
                dci.boundary_fn = dci.skgtn_dst_pred_in.mean() if len(dci.skgtn_dst_pred_in) > 0 else 0
                dci.boundary_tp = max(0, 1 - dci.boundary_fp - dci.boundary_fn)

                dci.volume_gt = dci.component_gt.sum() * helper['voxel_volume']
                dci.volume_pred = dci.component_pred.sum() * helper['voxel_volume']

                dci.volume_tp = (dci.component_pred & dci.component_gt).sum() * helper['voxel_volume']
                dci.volume_fn = dci.volume_gt - dci.volume_tp
                dci.volume_fp = dci.volume_pred - dci.volume_tp

                dci.volume_tp_rate = dci.volume_tp / dci.volume_gt
                dci.volume_fn_rate = dci.volume_fn / dci.volume_gt if dci.volume_gt > 0 else 0
                dci.volume_fp_rate = dci.volume_fp / dci.volume_gt

                m = copy.deepcopy(m_def)

                m['D']['tp'] += dci.volume_tp_rate > alpha1
                m['D']['fn'] += 1 - (dci.volume_tp_rate > alpha1)
                m['D']['fp'] += dci.volume_fp_rate > alpha2

                m['U']['tp'] += len(dci.pred_comp) == 1
                m['U']['fn'] += len(dci.pred_comp) > 1

                m['T']['tp'] += dci.volume_tp
                m['T']['fn'] += dci.volume_fn
                m['T']['fp'] += dci.volume_fp

                m['R']['tp'] += dci.volume_tp_rate
                m['R']['fn'] += dci.volume_fn_rate
                m['R']['fp'] += min(1, dci.volume_fp_rate)

                m['B']['tp'] += dci.boundary_tp
                m['B']['fn'] += dci.boundary_fn
                m['B']['fp'] += dci.boundary_fp

                for x in resc['total']:
                    for y in resc['total'][x]:
                        resc['total'][x][y] += m[x][y]

                resc['components'][i] = {
                    'MME': m,
                    'detected': sum(dci.pred_comp) > 0,
                    'uniform_gt': (1. / len(dci.pred_comp)) if len(dci.pred_comp) > 0 else 0,
                    'uniform_pred': (1. / len(dci.rel_gts)) if len(dci.rel_gts) > 0 else 0,
                    # 'maxd': dci.max_dst_gt,
                    'hd': dci.hd,
                    'hd_avg': dci.hd_avg,
                    'hd_95': dci.hd95,
                    'hd gt2pred': dci.dst_border_gt2pred_abs.mean() if len(dci.dst_border_gt2pred) else 0,
                    'hd pred2gt': dci.dst_border_pred2gt_abs.mean() if len(dci.dst_border_pred2gt) else 0,
                    # 'hdn': self.info(dci.pred_dst / dci.max_dst_gt),
                    'skgtn': dci.skgtn_dst_pred.mean() if len(dci.skgtn_dst_pred) else 0,
                    'skgtn_tp': 1 - (np.clip(dci.skgtn_dst_pred, 0, 1).mean() if len(dci.skgtn_dst_pred) else 0),
                    'skgtn_fn': dci.skgtn_dst_pred_in.mean() if len(dci.skgtn_dst_pred_in) else 0,
                    'skgtn_fp': dci.skgtn_dst_pred_out.mean() if len(dci.skgtn_dst_pred_out) else 0
                }
            resc['pN'] = dc.pN
            resc['gN'] = dc.gN

            # print(res)
            #     border_dst_shape=np.zeros(component_gt.shape,bool)
            #     border_dst_shape[border_pred]=dst[border_pred]
            #     plt.imshow(border_dst_shape)

            # print(dst[border_pred])

            # print(dst[0,0])
            dc.prs = {}
            for i in range(1, dc.pN + 1):
                dc.prs[i] = dci = common.Object()
                dci.component_p = dc.pred_labels == i
                gt_labels = dc.helperc['gt_labels']
                dci.rel_gt_comps, dci.rel_gts = _get_component_of(gt_labels, gt_labels[dci.component_p], dc.gN)

                if len(dci.rel_gts) == 0:
                    resc['total']['D']['fp'] += 1
                resc['total']['U']['fp'] += len(dci.rel_gts) > 1
        if return_debug_data:
            return res, data
        return res

    # def info(self, na):
    #     return {
    #         'avg': na.mean() if len(na) > 0 else np.nan,
    #         'min': na.min() if len(na) > 0 else np.nan,
    #         '25': np.quantile(na, 0.25) if len(na) > 0 else np.nan,
    #         '50': np.quantile(na, 0.5) if len(na) > 0 else np.nan,
    #         '75': np.quantile(na, 0.75) if len(na) > 0 else np.nan,
    #         '95': np.quantile(na, 0.95) if len(na) > 0 else np.nan,
    #         'max': na.max() if len(na) > 0 else np.nan
    #     }


def _get_component_of(img, classes, max_component=None):
    idx = np.s_[:]  #geometry.one_roi(img, return_index=True)
    idx2 = np.s_[:]  #geometry.one_roi(classes, return_index=True)
    max_component = max_component or classes.max()

    pred_comp = [c for c in range(1, max_component + 1) if c in classes[idx2]]

    component_pred = np.zeros(img.shape, bool)

    for l in pred_comp:
        component_pred[idx] |= img[idx] == l

    return component_pred, pred_comp
