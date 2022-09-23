import skimage
from auto_profiler import Timer
from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion
import cc3d
import hashlib
import myutils
import edt
from .. import geometry
import copy
from eval_seg.common
class MME(Metric):

    def __init__(self, num_classes, debug={}):
        super().__init__(num_classes, debug)
        pass

    
    def set_reference(self,reference, spacing=None)
        self.reference=reference
        if spacing is None:
            spacing=[1,1,1]

        self.spacing=spacing
        self.helper=MME.calculate_info(reference,spacing,self.num_classes)
    
    @Cache.memoize()
    @staticmethod
    def calculate_info(cls,reference, spacing=None,num_classes=2)
        
        helper = {}
        
        helper['voxel_volume']=spacing[0]*spacing[1]*spacing[2]
        helper['class'] = {}
        for c in range(self.num_classes):
            print(f'class={c}')
    
            refc = reference == c

            gt_labels, gN = cc3d.connected_components(refc, return_N=True)
            helperc = {}
            helper['class'][c] = helperc
            helperc['gt_labels'] = gt_labels
            helperc['gN'] = gN
            
            gt_regions=geometry.expand_labels(gt_labels,spacing=spacing)
            
            helperc['components'] = {}
            for i in range(0, gN + 1):
                gt_component = gt_labels == i
                gt_region=gt_regions==i
                gt_border = geometry.find_binary_boundary(gt_component)
                
                in_dst = geometry.distance(gt_component,spacing=spacing,mode='in')
                out_dst = geometry.distance(gt_component,spacing=spacing,mode='out')
                gt_dst = out_dst+in_dst

                skeleton = geometry.skeletonize(gt_component,spacing=spacing) > 0
                skeleton_dst = geometry.distance(skeleton,spacing=spacing,mode='out')

                normalize_dst_inside = in_dst / (skeleton_dst + in_dst)
                normalize_dst_outside = np.maximum(0,out_dst - epsilon / (skeleton_dst - out_dst + epsilon))
                # normalize_dst_outside = normalize_dst_outside.clip(0, normalize_dst_outside.max())
                normalize_dst = normalize_dst_inside + normalize_dst_outside

                
                helperc['components'][i] = {
                    'gt': gt_component,
                    'gt_region':gt_region,
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
  


    def evaluate(self, test):
        reference=self.reference
        helper=self.helper
        assert test.shape==reference.shape,'reference and test are not match'
        
        
        
        alpha1=.1
        alpha2=1
        m_def={    'D':{'tp':0,'fp':0,'fn':0,'tn':0},
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

        res = {}
        
        data={}
        
        for c in range(self.num_classes):
            data[c]=dc=common.Object()
            
            dc.testc = test == c
            dc.helperc = helper['class'][c]

            res[c] = resc = {}
            
            dc.gt_labels, dc.gN = dc.helperc['gt_labels'], dc.helperc['gN']
            dc.pred_labels, dc.pN = cc3d.connected_components(dc.testc, return_N=True)
            
            resc['total']=copy.deepcopy(m_def)
            dc.gts={}
            for i in range(1, gN + 1):
                dc.gts[i]=dci=common.Object()
                hci = helperc['components'][i]
                dci.component_gt = hci['gt'] 
                dci.component_pred, dci.pred_comp=_get_component_of(dc.pred_labels,dc.pred_labels[dci.component_gt])
                
                dci.rel_gt_comps,dci.rel_gts=_get_component_of(dc.gt_labels,dc.gt_labels[dci.component_pred])

                dci.pred_in_region=dci.component_pred
                # if a prediction contains two gt only consider the part related to gt
                for l in rel_gts:
                    if l!=i:
                        dci.pred_in_region=dci.pred_in_region& ~helperc['components'][l]['gt_region']
                        
                
                
                dci.border_gt = hci['gt_border']
                dci.border_pred = geometry.find_binary_boundary(dci.pred_in_region)
                 
                dci.dst_border_gt2pred = hci['gt_dst'][dci.border_pred]
                dci.gt_hd = dci.dst_border_gt2pred.max() if len(dci.dst_border_gt2pred) > 0 else np.nan
                dci.gt_hd_avg = dci.dst_border_gt2pred.mean() if len(dci.dst_border_gt2pred) > 0 else np.nan
                dci.gt_hd95 = np.quantile(dci.dst_border_gt2pred, 0.95) if len(dci.dst_border_gt2pred) > 0 else np.nan
                
                dci.pred_border_dst=geometry.distance(dci.component_pred,
                                                      mode='both',
                                                      mask=dci.rel_gt_comps|dci.component_pred)
                
                dci.dst_border_pred2gt = dci.pred_border_dst[dci.border_gt]
                dci.pred_hd = dci.dst_border_pred2gt.max() if len(dci.dst_border_pred2gt) > 0 else np.nan
                dci.pred_hd_avg = dci.dst_border_pred2gt.mean() if len(dci.dst_border_pred2gt) > 0 else np.nan
                dci.pred_hd95 = np.quantile(dci.dst_border_pred2gt, 0.95) if len(dci.dst_border_pred2gt) > 0 else np.nan
                
                dci.hd=np.mean([dci.gt_hd,dci.pred_hd])
                dci.hd_avg=np.mean([dci.gt_hd_avg,dci.pred_hd_avg])
                dci.hd95=np.mean([dci.gt_hd95,dci.pred_hd95])

                dci.skgtn_dst_in = hci['skgt_normalized_dst_in']
                dci.skgtn_dst_pred_in = dci.skgtn_dst_in[dci.border_pred]
                dci.skgtn_dst_pred_in = dci.skgtn_dst_pred_in[dci.skgtn_dst_pred_in > 0]

                dci.skgtn_dst_out = hci['skgt_normalized_dst_out']
                dci.skgtn_dst_pred_out = dci.skgtn_dst_out[dci.border_pred]
                dci.skgtn_dst_pred_out = dci.skgtn_dst_pred_out[dci.skgtn_dst_pred_out > 0]

                dci.skgtn_dst_pred = np.concatenate([dci.skgtn_dst_pred_out, dci.skgtn_dst_pred_in])
                dci.skgtn_dst_pred = dci.skgtn_dst_pred[dci.skgtn_dst_pred > 0]
                
                
                
                dci.boundary_fp=min(1,dci.skgtn_dst_pred_out.mean()) if len(dci.skgtn_dst_pred_out)>0 else 0
                dci.boundary_fn=dci.skgtn_dst_pred_in.mean() if len(dci.skgtn_dst_pred_in)>0 else 0
                dci.boundary_tp=max(0,1-dci.boundary_fp-dci.boundary_fn)
                
                
                dci.volume_gt=dci.component_gt.sum()*helper['voxel_volume']
                dci.volume_pred=dci.component_pred.sum()*helper['voxel_volume']
                
                dci.volume_tp=(dci.component_pred & dci.component_gt).sum()*helper['voxel_volume']
                dci.volume_fn= dci.volume_gt-dci.volume_tp
                dci.volume_fp= dci.volume_pred-dci.volume_tp
                
                dci.volume_tp_rate=dci.volume_tp/dci.volume_gt
                dci.volume_fn_rate=dci.volume_fn/dci.volume_gt
                dci.volume_fp_rate=dci.volume_fp/dci.volume_gt
                
                m=copy.deepcopy(m_def)
                
                m['D']['tp']+=dci.volume_tp_rate>alpha1
                m['D']['fn']+=1-(dci.volume_tp_rate>alpha1)
                m['D']['fp']+=dci.volume_fp_rate>alpha2 #todo bug if it has overlap with multiple gt
                
                
                
                m['U']['tp']+=len(dci.pred_comp)==1
                m['U']['fn']+=len(dci.pred_comp)>1
                
                m['T']['tp']+=dci.volume_tp
                m['T']['fn']+=dci.volume_fn
                m['T']['fp']+=dci.volume_fp
                
                m['R']['tp']+=dci.volume_tp_rate
                m['R']['fn']+=dci.volume_fn_rate
                m['R']['fp']+=min(1,dci.volume_fp_rate)
                
                m['B']['tp']+=dci.boundary_tp
                m['B']['fn']+=dci.boundary_fn
                m['B']['fp']+=dci.boundary_fp
                
                for x in resc['total']:
                    for y in resc['total'][x]:
                        resc['total'][x][y]+=m[x][y]
                    
                resc[i] = {
                    'MME':m,   
                    'detected': sum(dci.pred_comp) > 0,
                    'uniform_gt': 1. / len(dci.pred_comp) if len(dci.pred_comp) > 0 else 0,
                    'uniform_pred': 1. / len(dci.rel_gts) if len(dci.rel_gts) > 0 else 0,
                    'maxd': dci.max_dst_gt,
                    'hd': self.info(dci.pred_dst),
                    'hdn': self.info(dci.pred_dst / dci.max_dst_gt),
                    'skgtn': self.info(dci.skgtn_dst_pred),
                    'skgtn_tp': self.info(1 - np.clip(dci.skgtn_dst_pred, 0, 1)),
                    'skgtn_fn': self.info(dci.skgtn_dst_pred_in),
                    'skgtn_fp': self.info(dci.skgtn_dst_pred_out)
                }
            res[c]={
                **res[c],
                'pN': dc.pN,
                'gN': dc.gN,
                
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

    
def _get_component_of(img,classes,ignore_zero=True):
    pred_comp = np.unique(classes)
    if ignore_zero:
        pred_comp = pred_comp[pred_comp > 0] # remove zeros

    component_pred = np.zeros(img.shape, bool)

    for l in pred_comp:
        component_pred |= img == l
        
    return img,pred_comp