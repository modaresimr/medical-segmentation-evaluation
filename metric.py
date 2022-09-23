import numpy as np

epsilon = 0.000000001
import medpy.metric


class Metric:

    def __init__(self, num_classes, debug={}):
        self.num_classes = num_classes
        self.debug = debug
        pass

    def precompute(self, reference, spacing=None, connectivitiy=1):
        pass

    def evaluate(self, gt, pred, helper=None):
        pass


class PixelBasedCM(Metric):

    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    def precompute(self, reference, spacing=None, connectivitiy=1):
        pass

    def evaluate(self, gt, pred, helper=None):
        res = {
            'tp': np.zeros(self.num_classes, dtype=int),
            'fn': np.zeros(self.num_classes, dtype=int),
            'fp': np.zeros(self.num_classes, dtype=int),
            'tn': np.zeros(self.num_classes, dtype=int),
            'dice': np.zeros(self.num_classes, dtype=float),
            'accuracy': np.zeros(self.num_classes, dtype=float),
            'iou': np.zeros(self.num_classes, dtype=float),
            'precision': np.zeros(self.num_classes, dtype=float),
            'recall': np.zeros(self.num_classes, dtype=float),
            'fscore': np.zeros(self.num_classes, dtype=float),
            'fpr': np.zeros(self.num_classes, dtype=float),
        }
        total = (gt >= 0).sum()
        for c in range(self.num_classes):
            gtc = gt == c
            prc = pred == c
            total_gt = gtc.sum()
            total_pr = prc.sum()
            tp = (gtc & prc).sum()
            fn = total_gt - tp
            fp = total_pr - tp
            tn = total - tp - fn - fp
            res['tp'][c] = tp
            res['fn'][c] = fn
            res['fp'][c] = fp
            res['tn'][c] = tn

            res['accuracy'][c] = float((tp + tn) / (tp + fp + tn + fn + epsilon))
            res['dice'][c] = float(2. * tp / (2 * tp + fp + fn + epsilon))
            res['iou'][c] = float(tp / (tp + fp + fn + epsilon))
            prc = float(tp / (tp + fp + epsilon))
            res['precision'][c] = prc
            recall = float(tp / (tp + fn + epsilon))
            res['recall'][c] = recall
            res['fscore'][c] = 2 * prc * recall / (prc + recall + epsilon)
            res['fpr'][c] = float(fp / (tn + fp + epsilon))

        return res


class Other(Metric):

    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    def precompute(self, reference, spacing=None, connectivitiy=1):
        pass

    def evaluate(self, reference, test, helper=None, spacing=None, connectivitiy=1):
        connectivity = 1
        res = {}
        res['hd'] = medpy.metric.hd(test, reference, spacing, connectivitiy)  #Hausdorff Distance
        res['hd95'] = medpy.metric.hd95(test, reference, spacing, connectivitiy)  #Hausdorff Distance 95
        res['asd'] = medpy.metric.asd(test, reference, spacing, connectivitiy)  #Avg. Surface Distance
        res['assd'] = medpy.metric.assd(test, reference, spacing, connectivitiy)  #Avg. Symmetric Surface Distance
        import nnunet.evaluation.surface_dice

        res['nsd'] = nnunet.evaluation.surface_dice.normalized_surface_dice(test, reference, 1, spacing, connectivitiy)
        return res

    
class MISeval(Metric):
    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        
        self.metrics = ['TP', 'FP', 'FN','TN',
                        'TPR','Precision','TNR',
                        'Accuracy','BalancedAccuracy',
                        'Dice',
                        'IoU',
                        'AUC',
                        # 'HD', 'AHD',
                        'BoundaryDistance',
                        'Kappa',
                        'RandIndex','AdjustedRandIndex',
                        'Hinge',
                        'VolumetricSimilarity', 
                        'CrossEntropy', 
                        'MCC', 'MCC_normalized','MCC_absolute'  ]
        pass

    def precompute(self, reference):
        pass
    
    def evaluate(self, reference, test, helper=None):
        res={}
        import miseval

        for m in self.metrics:
            try:
                res[m] = miseval.evaluate(reference, test, metric=m, multi_class=True, n_classes=self.num_classes)    
            except:
                res[m]=None
                if self.debug.get('error',0):
                    raise
                

        return res
            
import auto_profiler


class MME(Metric):

    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    def precompute(self, reference, spacing=None, connectivitiy=1):
        import skimage
        from auto_profiler import Timer
        from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion
        import cc3d
        helper = {}
        # footprint = generate_binary_structure(reference.ndim, connectivity=1)
        # helper['footprint'] = footprint
        helper['class'] = {}
        for c in range(self.num_classes):
          print(c)
          if 1:# with Timer.instance(f'class {c}'):
            refc = reference == c
            if 1:#with Timer.instance(f'connected_components'):
                gt_labels, gN = cc3d.connected_components(refc, return_N=True)
            helperc = {}
            helper['class'][c] = helperc
            helperc['gt_labels'] = gt_labels
            helperc['gN'] = gN
            
            if 1:#with Timer.instance(f'region watershed'):
                # if c==0:
                    gt_regions=np.ones(gt_labels.shape)
                # else:
                    # out_dst = distance_transform_edt(gt_labels==0,sampling=spacing)
                    # gt_regions=skimage.segmentation.watershed(out_dst, gt_labels)
            # continue
            helperc['components'] = {}
            for i in range(0, gN + 1):
              if 1:#with Timer.instance(f'component{i}'):
                gt_component = gt_labels == i
                gt_region=gt_regions==i
                # gt_border = ~binary_erosion(gt_component, structure=footprint, iterations=1) & (gt_component)
                if 1:#with Timer.instance(f'boundary'):
                    gt_border = skimage.segmentation.find_boundaries(gt_component, connectivity=1, mode='thick', background=0)
                if 1:#with Timer.instance(f'distances'):
                    gt_border_dst = distance_transform_edt(~gt_border,sampling=spacing)
                    in_dst = distance_transform_edt(gt_component,sampling=spacing)
                    out_dst = distance_transform_edt(~(gt_component),sampling=spacing)

                if 1:#with Timer.instance('skeleton'):
                    skeleton = skimage.morphology.skeletonize(gt_component) > 0
                    skeleton_dst = distance_transform_edt(~skeleton)

                    normalize_dst_inside = in_dst / (skeleton_dst + in_dst)
                    normalize_dst_outside = out_dst - epsilon / (skeleton_dst - out_dst + epsilon)
                    normalize_dst_outside = normalize_dst_outside.clip(0, normalize_dst_outside.max())
                    normalize_dst = normalize_dst_inside + normalize_dst_outside

                
                helperc['components'][i] = {
                    'gt': gt_component,
                    'gt_region':gt_region,
                    'gt_border': gt_border,
                    'gt_border_dst': gt_border_dst,
                    'gt_out_dst': out_dst,
                    'gt_in_dst': in_dst,
                    'gt_skeleton': skeleton,
                    'gt_skeleton_dst': skeleton_dst,
                    'skgt_normalized_dst': normalize_dst,
                    'skgt_normalized_dst_in': normalize_dst_inside,
                    'skgt_normalized_dst_out': normalize_dst_outside
                }
                if self.debug.get('show_precompute',0):
                    self.debug_helper(helperc['components'][i])

        return helper
    


    def debug_helper(self,helpercc):
        f={}
        for k in helpercc:
            x=helpercc[k]
            x[x>10]=0
            f[k]= np.clip(x,0,5)/min(5,x.max())
        import plotly.express as px
        import myutils
        data=np.array(list(f.values()))
        data=myutils.array_trim(data,ignore=[0])
        fig=px.imshow(data,animation_frame=3,facet_col=0, facet_col_wrap=5 )
        itemsmap={f'{i}':key for i, key in enumerate(f)}
        fig.for_each_annotation(lambda a: a.update(text=itemsmap[a.text.split("=")[1]]))
        # fig.write_html('a.html')
        fig.show()
        
    def find_holes(self,image):
        from skimage.morphology import remove_small_holes
        import cc3d
        total_area=img.sum()
        without_holes= remove_small_holes(image, total_area)
        holes=without_holes^image
        labels, gN = cc3d.connected_components(holes, return_N=True)
        
        return holes
        

        
    def evaluate(self, reference, test, helper=None, spacing=None, connectivitiy=1):
        alpha1=.1
        alpha2=1
        m_def={'D':{'tp':0,'fp':0,'fn':0,'tn':0},
                   'B':{'tp':0,'fp':0,'fn':0,'tn':0},
                   'U':{'tp':0,'fp':0,'fn':0,'tn':0},
                   'R':{'tp':0,'fp':0,'fn':0,'tn':0},
                   'T':{'tp':0,'fp':0,'fn':0,'tn':0}
                  }
        
        dv = self.debug.get('v', False)
        print(f'dv-{dv}')
        if dv:
            print(f'dv-->True')
            vis = {}
        import cc3d
        import copy
        from scipy.ndimage import generate_binary_structure, binary_erosion, distance_transform_edt
        if helper == None:
            helper = self.precompute(reference)
        res = {}
        
        for c in range(self.num_classes):
            helperc = helper['class'][c]
            
            resc = {}
            res[c] = resc
            
            gt_labels, gN = helperc['gt_labels'], helperc['gN']
            
            #cc3d.connected_components(reference, return_N=True)
            testc = test == c
            pred_labels, pN = cc3d.connected_components(testc, return_N=True)

            import matplotlib.pyplot as plt
            import plotly.express as px

            # if self.debug.get('view3d',0):

            # plt.subplot(1,2,1)
            # plt.imshow(gt_labels)
            # plt.subplot(1,2,2)
            # plt.imshow(pred_labels)
            # plt.show()
            # footprint = generate_binary_structure(reference.ndim, connectivity=1)
            resc['total']=copy.deepcopy(m_def)
            for i in range(1, gN + 1):
                component = helperc['components'][i]
                component_gt = component['gt']  #gt_labels==i
                if dv: vis[f'component{i}_gt'] = component['gt']
                rel_pred = pred_labels[component_gt]
                pred_comp = np.unique(rel_pred)
                pred_comp = pred_comp[pred_comp > 0]
                print('pred_comp', pred_comp)
                component_pred = np.zeros(component_gt.shape, bool)
                
                for l in pred_comp:
                    component_pred |= pred_labels == l
                if dv:
                    vis[f'component{i}_pred'] = component_pred

                rel_gts = np.unique(component_gt[component_pred])
                rel_gts = rel_gts[rel_gts > 0]

                # print(pred_comp)
                # border_gt=~binary_erosion(component_gt, structure=footprint, iterations=1)& (component_gt)
                border_gt = component['gt_border']
                border_pred = skimage.segmentation.find_boundaries(component_pred, connectivity=1, mode='thick', background=0)
                # border_pred = ~binary_erosion(component_pred, structure=footprint, iterations=1) & (component_pred)
                if dv:
                    vis[f'component{i}_border_gt'] = border_gt
                    vis[f'component{i}_border_pred'] = border_pred
                dst = component['gt_border_dst']  #distance_transform_edt(~border_gt)

                # plt.subplot(1,N+1,i)
                # plt.imshow(dst)
                #     print(dst[component_gt])
                max_dst_gt = dst[component_gt].max() if len(dst[component_gt]) > 0 else np.nan
                pred_dst = dst[border_pred]

                hd = pred_dst.max() if len(pred_dst) > 0 else np.nan
                hd_avg = pred_dst.mean() if len(pred_dst) > 0 else np.nan
                hd95 = np.quantile(pred_dst, 0.95) if len(pred_dst) > 0 else np.nan

                skgtn_dst_in = component['skgt_normalized_dst_in']
                skgtn_dst_pred_in = skgtn_dst_in[~component_pred]
                if dv:
                    vis[f'component{i}_skeleton_gt'] = component['gt_skeleton']
                    vis[f'component{i}_skeleton_gt_dst'] = component['gt_skeleton_dst']
                    vis[f'component{i}_skgtn_dst_pred_in'] = np.zeros(skgtn_dst_in.shape)
                    vis[f'component{i}_skgtn_dst_pred_in'][~component_pred] = skgtn_dst_in[~component_pred]

                skgtn_dst_pred_in = skgtn_dst_pred_in[skgtn_dst_pred_in > 0]

                skgtn_dst_out = component['skgt_normalized_dst_out']
                skgtn_dst_pred_out = skgtn_dst_out[component_pred]
                if dv:
                    vis[f'component{i}_skgtn_dst_pred_out'] = np.zeros(skgtn_dst_out.shape)
                    vis[f'component{i}_skgtn_dst_pred_out'][component_pred] = skgtn_dst_out[component_pred]

                skgtn_dst_pred_out = skgtn_dst_pred_out[skgtn_dst_pred_out > 0]

                # skgtn_dst = component['skgt_normalized_dst']
                skgtn_dst_pred = np.concatenate([skgtn_dst_pred_out, skgtn_dst_pred_in])
                skgtn_dst_pred = skgtn_dst_pred[skgtn_dst_pred > 0]
                if dv:
                    vis[f'component{i}_skgtn_dst_pred'] = vis[f'component{i}_skgtn_dst_pred_in'] + vis[f'component{i}_skgtn_dst_pred_out']
                
                boundary_fp=min(1,skgtn_dst_pred_out.mean()) if len(skgtn_dst_pred_out)>0 else 0
                boundary_fn=skgtn_dst_pred_in.mean() if len(skgtn_dst_pred_in)>0 else 0
                boundary_tp=1-boundary_fp-boundary_fn
                
                
                surface_gt=component_gt.sum()
                surface_pred=component_pred.sum()
                
                surface_tp=(component_pred & component_gt).sum()
                surface_fn= surface_gt-surface_tp
                surface_fp= surface_pred-surface_tp
                
                
                
                
                
                surface_tp_rate=surface_tp/surface_gt
                surface_fn_rate=surface_fn/surface_gt
                surface_fp_rate=surface_fp/surface_gt
                
                m=copy.deepcopy(m_def)
                
                m['D']['tp']+=surface_tp_rate>alpha1
                m['D']['fn']+=1-(surface_tp_rate>alpha1)
                m['D']['fp']+=surface_fp_rate>alpha2 #todo bug if it has overlap with multiple gt
                
                
                
                m['U']['tp']+=len(pred_comp)==1
                m['U']['fn']+=len(pred_comp)>1
                
                m['T']['tp']+=surface_tp
                m['T']['fn']+=surface_fn
                m['T']['fp']+=surface_fp
                
                m['R']['tp']+=surface_tp_rate
                m['R']['fn']+=surface_fn_rate
                m['R']['fp']+=min(1,surface_fp_rate)
                
                m['B']['tp']+=boundary_tp
                m['B']['fn']+=boundary_fn
                m['B']['fp']+=boundary_fp
                for x in resc['total']:
                    for y in resc['total'][x]:
                        resc['total'][x][y]+=m[x][y]
                    
                resc[i] = {
                    'MME':m,
                    'pN': pN,
                    'gN': gN,
                    'detected': sum(pred_comp) > 0,
                    'uniform_gt': 1. / len(pred_comp) if len(pred_comp) > 0 else 0,
                    'uniform_pred': 1. / len(rel_gts) if len(rel_gts) > 0 else 0,
                    'maxd': max_dst_gt,
                    'hd': self.info(pred_dst),
                    'hdn': self.info(pred_dst / max_dst_gt),
                    'skgtn': self.info(skgtn_dst_pred),
                    'skgtn_tp': self.info(1 - np.clip(skgtn_dst_pred, 0, 1)),
                    'skgtn_fn': self.info(skgtn_dst_pred_in),
                    'skgtn_fp': self.info(skgtn_dst_pred_out)
                }
                
                # print(res)
            #     border_dst_shape=np.zeros(component_gt.shape,bool)
            #     border_dst_shape[border_pred]=dst[border_pred]
            #     plt.imshow(border_dst_shape)

            # print(dst[border_pred])

            # print(dst[0,0])
            for i in range(1, pN + 1):              
                component_p = pred_labels==i
                if dv: vis[f'component{i}_p']=component_p
                gt_labels=helperc['gt_labels']
                rel_gt = gt_labels[component_p]
                unique_rel_gt_labels=np.unique(rel_gt[rel_gt>0])
                if len(rel_gt)==0:
                    resc['total']['D']['fp']+=1
                resc['total']['U']['fp']+=len(unique_rel_gt_labels)>1
            if dv:
                for k in vis:
                    vis[k][vis[k] > 10] = 0
                    vis[k] = np.clip(vis[k], 0, 5) / max(1, min(5, vis[k].max()))

                data = np.array(list(vis.values()))

                import myutils
                data = myutils.array_trim(data, ignore=[0])
                fig = px.imshow(
                    data,
                    animation_frame=3,
                    facet_col=0,
                )
                for i, key in enumerate(vis.keys()):
                    fig.layout.annotations[i]['text'] = key
                if dv == 'show':
                    fig.show()
                else:
                    fig.write_html(dv.replace('.html', f'-{c}.html'))
        return res

    def info(self, na):
        return {
            'avg': na.mean() if len(na) > 0 else np.nan,
            'min': na.min() if len(na) > 0 else np.nan,
            '25': np.quantile(na, 0.25) if len(na) > 0 else np.nan,
            '50': np.quantile(na, 0.5) if len(na) > 0 else np.nan,
            '75': np.quantile(na, 0.75) if len(na) > 0 else np.nan,
            '95': np.quantile(na, 0.95) if len(na) > 0 else np.nan,
            'max': na.max() if len(na) > 0 else np.nan
        }
