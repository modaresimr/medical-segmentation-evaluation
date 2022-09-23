import k3d
import CTHelper
import numpy as np
from tqdm.auto import tqdm
import myutils
import skimage
import matplotlib.pyplot as plt
color_map = [
    '#f884f7', '#f968c4', '#ea4388', '#cf244b', '#b51a15', '#bd4304', '#cc6904', '#d58f04', '#cfaa27', '#a19f62', '#588a93', '#2269c4', '#3e3ef0', '#6b4ef9',
    '#956bfa', '#cd7dfe', '#f884f7'
]
def colormap_convert(arr):
    arr=np.multiply(arr,255).astype(int)
    tall= [(arr[x]<<24)+(arr[x+1]<<16)+(arr[x+2]<<8)+(arr[x+3]) for x in range(0,len(arr),4)]
    # if indx==None: 
    return myutils.CircleList(tall)
    # return [tall[indx%len(tall)]])

import k3d

from k3d.colormaps import basic_color_maps,matplotlib_color_maps 
# colormap=cm(basic_color_maps.RainbowDesaturated)
cls_colormap=colormap_convert(matplotlib_color_maps.tab10)[::28]
tp_colormap=colormap_convert(matplotlib_color_maps.Greens)[28*3::28]
fn_colormap=colormap_convert(matplotlib_color_maps.Blues)[28*3::28]
fp_colormap=colormap_convert(matplotlib_color_maps.Reds)[28*3::28]

bone_colormap=colormap_convert(matplotlib_color_maps.bone)


def convert2dataframe(img):
    import pandas as pd
    names = ['x', 'y', 'z']

    index = pd.MultiIndex.from_product([range(s) for s in img.shape], names=names)
    df = pd.DataFrame({'value': img.flatten()}, index=index)['value']
    data = df[df > 0].reset_index()
    return data


def plot_3d_old(img, dst=None):
    data = convert2dataframe(img)

    import plotly.graph_objects as go
    import numpy as np
    X = data['x'].values
    Y = data['y'].values
    Z = data['z'].values
    values = data['value'].values

    fig = go.Figure(data=go.Mesh3d(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        # color=values.flatten(),

        # isomin=0,
        # isomax=2,
        # opacity=0.1, # needs to be small to see through all surfaces
        # surface_count=17, # needs to be a large number for good volume rendering
    ))

    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[0,gt.shape[0]],),
    #                      yaxis = dict(nticks=4, range=[0,gt.shape[1]],),
    #                      zaxis = dict(nticks=4, range=[0,gt.shape[2]],),),
    #     width=700,
    #     margin=dict(r=20, l=10, b=10, t=10))
    if dst == None:
        fig.show()
    else:
        fig.write_html(dst)


def plot_3d_heavy(img, dst=None):
    import plotly
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    import numpy as np
    # local helper function
    from plotui import mesh_cubes
    voxels = img
    x_max = voxels.shape[0]
    y_max = voxels.shape[1]
    z_max = voxels.shape[2]
    cubes = mesh_cubes(voxels)
    data = go.Data(cubes)
    layout = go.Layout(
        scene=dict(
            aspectmode='data',
            # aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(
                type='linear',
                range=[0, x_max],
                showgrid=False,
            ),
            yaxis=dict(
                type='linear',
                range=[0, y_max],
                showgrid=False,
            ),
            zaxis=dict(
                type='linear',
                range=[0, z_max],
                showgrid=False,
            ),
        ),)
    fig = go.Figure(data=data, layout=layout)

    # iplot(fig, filename='voxels')
    if dst == None:
        fig.show()
    else:
        fig.write_html(dst)


# functions for visualizing voxels as 3D cubes
import numpy as np
import plotly.graph_objs as go


def _to_cube(origin):
    '''Convert a voxel at origin into arrays of vertices and triangular faces.
    Args:
        origin - a 3-tuple, the origin of the cube
    Returns:
        (x, y, z, i, j, k) - tuple of list(int)
            x, y, z are the cube vertices
            i, j, k are the cube faces
    '''

    x, y, z = [], [], []
    for v in range(8):
        x.append(origin[0] + v // 4)
        y.append(origin[1] + v // 2 % 2)
        z.append(origin[2] + v % 2)

    i = [0, 1, 0, 1, 0, 6, 2, 7, 4, 7, 1, 7]
    j = [1, 2, 1, 4, 2, 2, 3, 3, 5, 5, 3, 3]
    k = [2, 3, 4, 5, 4, 4, 6, 6, 6, 6, 5, 5]

    return x, y, z, i, j, k


def mesh_cubes(vox_image: np.ndarray):
    '''Turn a 3d array into a list of plotly.go.Mesh3d objects
    Args:
        vox_image - a numpy.ndarray of binary values
    Returns:
        cubes - a list of cubes as Mesh3d objects
    '''

    # must be a 3-d array
    assert vox_image.ndim == 3

    cubes = []
    import plotly.express as px

    cm = px.colors.cyclical.mrybm
    import math

    # iterate over the entire 3d array
    for i_ind in range(vox_image.shape[0]):
        for j_ind in range(vox_image.shape[1]):
            for k_ind in range(vox_image.shape[2]):
                val = vox_image[i_ind, j_ind, k_ind]
                if val > 0:
                    # convert the voxel at this position into a cube
                    cube = _to_cube((i_ind, j_ind, k_ind))
                    # append these vertices and faces as a Mesh3d object
                    cubes.append(
                        go.Mesh3d(x=cube[0],
                                  y=cube[1],
                                  z=cube[2],
                                  i=cube[3],
                                  j=cube[4],
                                  k=cube[5],
                                  hoverinfo='none',
                                  color=cm[math.ceil(val)],
                                  opacity=1 - (math.ceil(val) - val)))

    return cubes


def plot_3d(gt, pred=None, dst=None):
    import ipyvolume as ipv

    fig = ipv.figure(width=800, height=600)

    def convert2dataframe(img):
        import pandas as pd
        names = ['x', 'y', 'z']

        index = pd.MultiIndex.from_product([range(s) for s in img.shape], names=names)
        df = pd.DataFrame({'value': img.flatten()}, index=index)['value']
        data = df[df > 0].reset_index()
        return data

    for c in range(1, int(gt.max() + 1)):
        gtc = gt == c
        prc = pred == c

        total_pr = prc
        tp = (gtc & prc)
        fn = gtc & ~tp
        fp = prc & ~tp

        if tp.sum() > 0:
            data = convert2dataframe(tp).astype(np.float64)
            s = ipv.scatter(data['x'], data['y'], data['z'], color=[0., 1.0, 0.], opacity=.4, size=1, marker='sphere')
        if fn.sum() > 0:
            data = convert2dataframe(fn).astype(np.float64)
            s = ipv.scatter(data['x'], data['y'], data['z'], color=[1., .0, .0], opacity=.4, size=1, marker='sphere')
        if fp.sum() > 0:
            data = convert2dataframe(fp).astype(np.float64)
            s = ipv.scatter(data['x'], data['y'], data['z'], color=[0., 0, 1.], opacity=.4, size=1, marker='sphere')

        # ipv.zlim(0,100)
        ipv.pylab.style.box_off()
        ipv.view(-30, 40)
        # ipv.show()
        ipv.pylab.save(dst.replace('.html', f'-{c}.html'))


def display_seg(data,args={}):
    import skimage
    import matplotlib.pyplot as plt
    if args.get('slices',0):
        data=data[:,:,args['slices']]
    # from skimage.util.montage import montage2d
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 20))
    ax1.imshow(skimage.util.montage(np.moveaxis(data,2,0)),padding_width=4, cmap='bone')
    # fig.savefig('ct_scan.png')
    fig.show()


def multi_plot(gt, preds, dst=None,args={}):
    if len(gt.shape)==2:
        gt=gt.reshape(gt.shape[0],gt.shape[1],1)
    for pr in preds:
        if len(preds[pr].shape)==2:
            preds[pr]=preds[pr].reshape(preds[pr].shape[0],preds[pr].shape[1],1)
    hasz=gt.shape[2]>2
    
    
    
    plot = k3d.plot(grid=[0, gt.shape[0], 0, gt.shape[1], 0, gt.shape[2]], name=dst, grid_auto_fit=False,camera_auto_fit=hasz)
    if dst==None:
        plot.display()
    if not hasz:
        plot.camera = [512, 200, 200, 0, 200, 200, 0, 0, 1]

    v= k3d.voxels(gt.astype(np.uint8), opacity=0.3, compression_level=9, name='gt', group='gt')
    v.visible=args.get('show_all_gt',0)
    v.outlines=False
    plot+=v
    # for c in range(1, pred.max() + 1):
    #     plot += k3d.voxels((gt == c).astype(np.uint8), opacity=0.0, color=color_map[c], compression_level=9, name=f'{c}', group='gt')
    for c in tqdm(range(1, int(gt.max() + 1)),leave=False):
        v= k3d.voxels((gt == c).astype(np.uint8), opacity=0.2, color_map=[cls_colormap[c]], compression_level=9, name=f'{c}', group='gt')
        v.visible=args.get('show_each_gt',{c:1}).get(c,0)
        v.outlines=False
        plot += v
        if args.get('calc_skeleton_gt',{}).get(c,0):
            shape=(gt == c).astype(np.uint8)
            from skimage.morphology import skeletonize,binary_closing
            import scipy
            from scipy import ndimage, misc
    #         smoothed = scipy.signal.medfilt (shape, 5)
            smoothed = ndimage.median_filter (shape, 5)
    #         smoothed=shape
    #         smoothed = skimage.filters.gaussian(shape)
            skeleton = binary_closing(skeletonize(smoothed))
    #         skeleton = skeletonize(shape)

            v= k3d.voxels(skeleton.astype(np.uint8), opacity=1, color_map=[cls_colormap[c]], compression_level=1, name=f'sk-{c}', group='gt')
            
            v.visible=args.get('show_skeleton_gt',{}).get(c,0)
            v.outlines=True
            plot += v

            v2= k3d.voxels(smoothed.astype(np.uint8), opacity=.2, color_map=[cls_colormap[c]], compression_level=1, name=f'smooth-{c}', group='gt')
            v2.visible=args.get('show_smooth_gt',{}).get(c,0)
            v2.outlines=False
            plot+=v2
        
    for p in tqdm(preds,leave=False):
        pred = preds[p].astype(np.uint8)
        fp = (gt != pred) & (pred>0)
        fn = (gt != pred) & (gt>0)
        tp = (gt == pred) & (gt>0)
        
        v=k3d.voxels(fp.astype(np.uint8), opacity=0.6, color_map=fp_colormap, compression_level=9, name='fp', group=p)
        v.visible=args.get('show_all_fp',0)
        v.outlines=False
        plot += v
        
        v= k3d.voxels(fn.astype(np.uint8), opacity=0.6, color_map=fn_colormap, compression_level=9, name='fn', group=p) #
        v.visible=args.get('show_all_fn',0)
        v.outlines=False
        plot += v
        
        
        v= k3d.voxels(tp.astype(np.uint8), opacity=0.2, color_map=tp_colormap, compression_level=9, name='tp', group=p)
        v.visible=args.get('show_all_tp',0)
        v.outlines=False
        plot +=v
        
        v= k3d.voxels(pred.astype(np.uint8) , opacity=0.2, compression_level=9, color_map=cls_colormap, name='pred', group=p)
        v.visible=args.get('show_all_pred',0)
        v.outlines=False
        plot +=v
        
        for c in tqdm(range(1, int(pred.max() + 1)),leave=False):
            v = k3d.voxels((pred == c).astype(np.uint8), opacity=0.2, color_map=[cls_colormap[c]], compression_level=9, name=f'{c}', group=p)
            v.visible=args.get('show_each_pred',{c:1}).get(c,0)
            v.outlines=False
            plot +=v
            
        
        # plot += k3d.voxels(pred * 3, opacity=0.3, compression_level=5, name='pred', group=p)
        # plot += k3d.voxels(pred * 3, opacity=0.3, compression_level=5, name=p)
        # for c in range(1, pred.max() + 1):
        #     plot += k3d.voxels((pred == c).astype(np.uint8), opacity=0.0, color=color_map[c + 5], compression_level=9, name=f'{c}', group=p)

    if dst != None:
        with open(dst, 'w') as fp:
            fp.write(plot.get_snapshot())


def plot_voxels(gt, dst=None,show=False):

    plot = k3d.plot(grid=[0, 512, 0, 512, 0, 512], name=dst, grid_auto_fit=False)

    plot += k3d.voxels(gt.astype(np.uint8), opacity=0.3, compression_level=9)
    # for c in range(1, pred.max() + 1):
    #     plot += k3d.voxels((gt == c).astype(np.uint8), opacity=0.0, color=color_map[c], compression_level=9, name=f'{c}', group='gt')

    if show:
        plot.display()
    if dst!=None:
        with open(dst, 'w') as fp:
            fp.write(plot.get_snapshot())
            
            
def plot_multi_slice_seg(gt,pred,dst=None,args={}):
    import myutils
    f={}
    
    for c in tqdm(range(1,gt.max()+1),leave=False):
        f['gt']=np.clip(gt,0,5)/min(5,gt.max())
        for p in pred:
            f[p]= np.clip(pred[p],0,5)/min(5,pred[p].max())

        data=np.array(list(f.values()))
        data=myutils.array_trim(data,ignore=[0])
        fig=px.imshow(data,animation_frame=3,facet_col=0, )
        for i, key in enumerate(keys):
            fig.layout.annotations[i]['text'] = key
        
        fig.set_tile(f'class {c}')
        if dst!=None:
            fig.write_html(f'{dst}-{c}.html')
        if args.get('show',1):
            fig.show()

            

def slice(data,dim,cuts):

    
    if dim==1:
        data=np.transpose(data,(2,0,1))
        # data=data[::-1,::-1,:]
    elif dim==0:
        data=np.transpose(data,(2,1,0))
        # data=data[::-1,::-1,:]
    else:
        data=np.transpose(data,(1,0,2))
        # data=data[:,::-1,:] 
        # data=data[::-1,:,:] 
        
    if not hasattr(cuts,'__len__') or len(cuts):
        data=data[:,:,cuts] 
        
    return data

def orthoSlicer(img,pred,cut):
    row=1
    col=3
    fig,axes=plt.subplots(row,col,figsize=(col*2,row*2),dpi=100)
    axes=axes.reshape(-1)
    if len(pred)==0:
        pred['data']=np.zeros(img.shape)
    
    for p in pred:
        for i in range(3):
            axes[i].imshow(slice(img,i,cut[i]),cmap='bone')
            predcut=slice(pred[p],i,cut[i])
            if predcut.sum()>0:
                axes[i].contour(predcut,cmap='bone')
            if i==0:
                axes[i].axhline(cut[2])
                axes[i].axvline(cut[1])
            if i==1:
                axes[i].axhline(cut[2])
                axes[i].axvline(cut[0])
            if i==2:
                axes[i].axhline(cut[1])
                axes[i].axvline(cut[0])
            # axes[i].invert_xaxis()
            axes[i].set_ylim(0,predcut.shape[0])
            axes[i].set_xlim(0,predcut.shape[1])
            # axes[i].invert_yaxis()
            

def plot_multi_slice_image(img,pred,dst=None,args={}):
    import myutils
    f={}
    imglbl=args.get('imglabel','img')
    origsize_lbl='orig_size'
    if args.get('add_notzoom_img',0):
        origsize_lbl=imglbl
        imglbl='Zoom to ROI'
    gtlbl='GroundTruth'
    
    items={imglbl:img,**pred}
    
    dim,cuts=args.get('slices',(2,[]))
    items={p:slice(items[p],dim,cuts) for p in items}
#     if anim_dim==1:
#         items={p:np.transpose(items[p],(2,0,1)) for p in items}
#         items={p:items[p][::-1,::-1,:] for p in items}
#     elif anim_dim==0:
#         items={p:np.transpose(items[p],(2,1,0)) for p in items}
#         items={p:items[p][::-1,::-1,:] for p in items}
#     else:
#         items={p:np.transpose(items[p],(1,0,2)) for p in items}
#         items={p:items[p][:,::-1,:] for p in items}
        
#     if len(ranges):
#         items={p:items[p][:,:,ranges] for p in items}
    
    
    print(items[imglbl].shape)
            
    if args.get('clahe',0):
        items[imglbl]=CTHelper.claheCT(items[imglbl])  
        
    if args.get('crop2roi',0):
        roi = CTHelper.getROI(items[imglbl],True)
        print(roi)
        items={p:items[p][roi] for p in items}  
    
    if args.get('zoom2segments',0):
        notzoom_img=items[imglbl]
        
        orig_ratio=notzoom_img.shape[1]/notzoom_img.shape[0]
        zoom_roi = CTHelper.get_segment_roi([items[p] for p in items if p !=imglbl],wh_ratio=orig_ratio)
        items={p:items[p][zoom_roi] for p in items}
        if args.get('add_notzoom_img',0):
            # items={p:CTHelper.upscale_ct(items[p],notzoom_img.shape) for p in items}
            items={origsize_lbl: notzoom_img ,**items }
            
        
        
      
    
        
#         if origsize_lbl in items:
#             items[origsize_lbl]=CTHelper.claheCT(items[origsize_lbl])  
        
    
    
        
    img=items[imglbl]
  
    
    gt=items[gtlbl]
    normalimg=(img-img.min())/(img.max()-img.min())
    
    data={}
    for p in tqdm(items,leave=False):        
        x=items[p]
        clipmin=x.min()
        clipmax=x.max() 
        
        data[p]={'pred':(np.clip(x,clipmin,clipmax)-clipmin)/(clipmax-clipmin+.0000000001)}#min(clipmax,items[p].max())}
        
        if p==origsize_lbl:
            pass
        elif p!=imglbl:
                fp = (gt != x) & (x>0)
                fn = (gt != x) & (gt>0)
                tp = (gt == x) & (gt>0)
                data[p]['fp']=fp
                data[p]['fn']=fn
                data[p]['tp']=tp
    
    

    
    mri_cmap='bone'#plotui.customMRIColorMapForMPL_TPFPFN()
    
    
    
    col=min(len(items),5)
    row=(len(items)-1)//col+1
    
    
    
    for anim in range(items[imglbl].shape[2]):
        
        # fig, ax1 = plt.subplots(1, 1, figsize=(row, col),dpi=100)
        # ,gridspec_kw={'left':0, 'right':0, 'top':0, 'bottom':0}
        fig,axes=plt.subplots(row,col,figsize=(col*2,row*2),dpi=100)
        fig.suptitle(f'frame: {cuts[anim]}')
        axes=axes.reshape(-1)
        for i,p in enumerate(data):
            current={d:data[p][d][:,:,anim] for d in data[p]}
            
            imgsize=20
            if p in [imglbl,origsize_lbl]:
                axes[i].imshow(current['pred'],cmap=mri_cmap,vmin=0, vmax=1,alpha=1 ,interpolation='nearest')    
                if p ==origsize_lbl:
                    from matplotlib.patches import Rectangle
                    y,x=zoom_roi[0].start,zoom_roi[1].start
                    h,w=zoom_roi[0].stop-y,zoom_roi[1].stop-x
                    
                    axes[i].add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor='blue',lw=2))
            else:
                from matplotlib.colors import ListedColormap, LinearSegmentedColormap
                if args.get('add_backimg',0) and not ():
                    axes[i].imshow(normalimg[:,:,anim]/2,cmap=mri_cmap,vmin=0, vmax=1,alpha=1 ,interpolation='nearest')
                if current['pred'].sum()>0:
                    color= 'lime' if p==gtlbl else 'yellow'
                    axes[i].contour(current['pred'],colors=color,alpha=1)
                if p!=gtlbl:
                    if 'tp' in current and current['tp'].sum()>0:
                        # axes[i].contour(current['tp'],corner_mask=False,cmap=ListedColormap([(0,0,0,0),'lime']),vmin=0, vmax=1, alpha=1 )
                        axes[i].imshow(current['tp'],cmap=ListedColormap([(0,0,0,0),'lime']),vmin=0, vmax=1, alpha=1 )
                    if 'fp' in current and current['fp'].sum()>0:
                        axes[i].contour(current['fp'],corner_mask=False,cmap=ListedColormap([(0,0,0,0),'yellow']),vmin=0, vmax=1, alpha=1 )
                        axes[i].imshow(current['fp'],cmap=ListedColormap([(0,0,0,0),'yellow']),vmin=0, vmax=1, alpha=1 )
                    if 'fn' in current and current['fn'].sum()>0:                        
                        axes[i].contour(current['fn'],corner_mask=False,cmap=ListedColormap([(0,0,0,0),'red']),vmin=0, vmax=1, alpha=1 )
                        axes[i].imshow(current['fn'],cmap=ListedColormap([(0,0,0,0),'red']),vmin=0, vmax=1, alpha=1 )
            axes[i].set_axis_off()
            axes[i].set_ylim(0,current['pred'].shape[0])
            axes[i].set_xlim(0,current['pred'].shape[1])
            axes[i].set_title(p)
        for j in range(i+1,len(axes)):
            fig.delaxes(axes[j])
        if args.get('show',1):
            fig.show()

        if dst!=None:
            fig.savefig(f'{dst}{anim}.png')
            
def plot_multi_slice_image_old(img,pred,dst=None,args={}):
    import myutils
    import plotly.express as px
    f={}
    imglbl=args.get('imglabel','img')
    origsize_lbl='orig_size'
    if args.get('add_notzoom_img',0):
        origsize_lbl=imglbl
        imglbl='Zoom to ROI'
    gtlbl='GroundTruth'
    items={imglbl:img,**pred}
    
    if args.get('slices',0):
        items={p:items[p][:,:,args['slices']] for p in items}
    if args.get('crop2roi',0):
        minx, miny, maxx, maxy = CTHelper.getROI(items[imglbl],True)
        print(minx, miny, maxx, maxy)
        items={p:items[p][miny:maxy, minx:maxx,:] for p in items}
    
    
    
    
    if args.get('zoom2segments',0):
        notzoom_img=items[imglbl]
        print(notzoom_img.shape)
        orig_ratio=notzoom_img.shape[1]/notzoom_img.shape[0]
        zoom_roi = CTHelper.get_segment_roi([items[p] for p in items if p !=imglbl],wh_ratio=orig_ratio)
        items={p:items[p][zoom_roi] for p in items}
        if args.get('add_notzoom_img',0):
            # items={p:CTHelper.upscale_ct(items[p],notzoom_img.shape) for p in items}
            items={origsize_lbl: notzoom_img ,**items }
            
        
        
      
    if args.get('clahe',0):
        items[imglbl]=CTHelper.claheCT(items[imglbl])  
        if origsize_lbl in items:
            items[origsize_lbl]=CTHelper.claheCT(items[origsize_lbl])  
        
    
    
        
    img=items[imglbl]
  
    
    gt=items[gtlbl]
    normalimg=(img-img.min())/(img.max()-img.min())
    
    data={}
    for p in tqdm(items,leave=False):        
        x=items[p]
        clipmin=x.min()
        clipmax=x.max() 
        
        # if args.get('autoclip',0):
        #     # x=np.clip(x,0,x.max())
        #     # x=np.power(x,5)
        #     clipmax=np.quantile(x,args.get('clipmax',.9))+1
        #     clipmin=np.quantile(x[x>x.min()],args.get('clipmin',.3))-1
        #     # clipmin=0
        data[p]={'pred':(np.clip(x,clipmin,clipmax)-clipmin)/(clipmax-clipmin+.0000000001)}#min(clipmax,items[p].max())}
        f[p]=.9* (np.clip(x,clipmin,clipmax)-clipmin)/(clipmax-clipmin+.0000000001)#min(clipmax,items[p].max())
        
        if p==origsize_lbl:
            border_size=4
            color=.97
            for i in range(border_size):
                for j in [-1,1]:
                    f[p][zoom_roi[0],zoom_roi[1].start+i*j,:]=color
                    f[p][zoom_roi[0],zoom_roi[1].stop+i*j,:]=color
                    f[p][zoom_roi[0].start+i*j,zoom_roi[1],:]=color
                    f[p][zoom_roi[0].stop+i*j,zoom_roi[1],:]=color
     
        elif p!=imglbl:
            
            if args.get('show_tp_fp_fn',0):
                fp = (gt != x) & (x>0)
                fn = (gt != x) & (gt>0)
                tp = (gt == x) & (gt>0)
                data[p]['fp']=fp
                data[p]['fn']=fn
                data[p]['tp']=tp
                
                f[p]=np.maximum(tp*.99,f[p])
                if p !=gtlbl:
                    f[p]=np.maximum(fp*.965,f[p])
                    f[p]=np.maximum(fn*.935,f[p])
                
            if args.get('add_backimg',0):
                f[p]=np.maximum(normalimg/2,f[p])
                
            
            
                
        
    
    if args.get('add_notzoom_img',0):
            f={p:CTHelper.upscale_ct(f[p],notzoom_img.shape) for p in f}

    data=np.array(list(f.values()))
    # data=myutils.array_trim(data,ignore=[0])
    

    # def getColorMap(gt):
    #     gt=(gt-gt.min())/(gt.max()-gt.min())
    #     gt=gt[gt>.1]#not include background
    #     gt=gt[gt>np.quantile(gt,.1)]
    #     # gt=gt[gt<np.quantile(gt,0.98)]
    #     mycolormap1=[[np.quantile(gt,i),f'rgb({int(i*192)},{int(i*192)},{int(i*192)})'] for i in np.arange (0,.9,.1)]
    #     mycolormap2=[[np.quantile(gt,.9+i/10),f'rgb({int((3+i)*64)},{int((3+i)*64)},{int((3+i)*64)})'] for i in np.arange (0,1,.1)]
    #     mycolormap=[[0,'rgb(0,0,0)'],*mycolormap1,*mycolormap2,[1,'rgb(255,255,255)']]
    #     return mycolormap

    mri_cmap=customMRIColorMapForMPL_TPFPFN()
    
    if args.get('interactive',0):
        fig=px.imshow(data, animation_frame=3, facet_col=0,
                            facet_col_wrap=min(len(items),5), zmin=0, zmax=1, 
                      # color_continuous_scale=customMRIColorMapForPlotlyTPFPFN(),
                      color_continuous_scale=matplotlib_to_plotly(mri_cmap),
                      
                      facet_col_spacing=0.01,facet_row_spacing =0.01
                      # width=1600, height=800
                     )
        itemsmap={f'{i}':key for i, key in enumerate(items)}
        fig.for_each_annotation(lambda a: a.update(text=itemsmap[a.text.split("=")[1]]))
        if args.get('show',1):
            fig.show()
        if dst!=None:
            fig.write_html(f'{dst}.html')
        
    else:
        import skimage
        import matplotlib.pyplot as plt
        col=min(len(items),5)
        row=(len(items)-1)//col+1
        pad=50
        newdata=np.zeros((data.shape[0],data.shape[1]+pad,data.shape[2],data.shape[3]))+.9
        newdata[:,pad:data.shape[1]+pad,:,:]=data
        
        for i,p in enumerate(items):
            for j in range(newdata.shape[3]):
            # newdata[i,0:20,:,:]=cv2.putText    
                putTextCenter(newdata[i,0:pad,:,j],p ,0.1)     
        
        for anim in range(data.shape[3]):
            montagimg=skimage.util.montage(newdata[:,:,:,anim], grid_shape=(row,col),padding_width=4,fill=0.9)
            # fig=px.imshow(montagimg,zmin=0, zmax=1, color_continuous_scale=customMRIColorMapForPlotlyTPFPFN())
            # fig.show()
            imgsize=20
            fig, ax1 = plt.subplots(1, 1, figsize=(imgsize*row, imgsize*col))
            ax1.imshow(montagimg,cmap=mri_cmap,vmin=0, vmax=1
                       ,interpolation='nearest'
                      )
            ax1.set_axis_off()

            if args.get('show',1):
                fig.show()
            
            if dst!=None:
                fig.savefig(f'{dst}.png')
            
def putTextCenter(img,text,color):
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # get boundary of this text
        textsize = cv2.getTextSize(text, font, 1, 5)[0]
        # get coords based on boundary
        textX = (img.shape[1] - textsize[0]) // 2
        textY = (img.shape[0] + textsize[1]) // 2
        

        # add text centered on image
        cv2.putText(img, text, (textX, textY ), font, 1, color, 2)
                    

def customMRIColorMapForMPL_TPFPFN():  
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        cmap=bone_colormap
        total=len(bone_colormap)/.9
        
        colors=[cmap[-1],0xff0000,0xFFFF00,0x00DD00]
        percolor=int((total-len(cmap))//len(colors))
        def int2rgb(rgbint):
            return (rgbint // 256 // 256 % 256)/256., (rgbint // 256 % 256)/256., (rgbint % 256)/256.,1
        print(percolor)
        newcmap=[*cmap,*[c for c in  colors for i in range(percolor) ]]
        return ListedColormap([int2rgb(c) for c in newcmap])
            
def matplotlib_to_plotly(cmap, pl_entries=255):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = (np.array(cmap(k*h)[:4])*255).astype(np.uint8)
        pl_colorscale.append([k*h, 'rgba'+str((C[0], C[1], C[2],C[3]))])

    return pl_colorscale

        
# def customMRIColorMapForPlotlyTPFPFN():  
#         def rgb2string(rgbint):
#             return f'rgb({rgbint // 256 // 256 % 256}, {rgbint // 256 % 256}, {rgbint % 256})'

#         colors=bone_colormap
#         cmap= [[i/(len(colors)/.9),rgb2string(colors[i])] for i in range(len(colors))]
#         return [*cmap,[.91,rgb2string(colors[-1])],[0.93,rgb2string(0xff0000)],[0.96,rgb2string(0xFFFF00)],[1,rgb2string(0x00BB00)]]
    
# def getMRIColorMapForPlotly():  
#         def rgb2string(rgbint):
#             return f'rgb({rgbint // 256 // 256 % 256}, {rgbint // 256 % 256}, {rgbint % 256})'

#         colors=bone_colormap
#         return [[i/(len(colors)-1),rgb2string(colors[i])] for i in range(len(colors))]
#         return[[0,rgb2string(colors[0])],*cmap,[1,rgb2string(colors[-1])]]
        
# def getMRIColorMapForPlotly(gt):
#         import plotui
#         import k3d
#         from k3d.colormaps import basic_color_maps,matplotlib_color_maps 
#         def rgb2string(rgbint):
#             return f'rgb({rgbint // 256 // 256 % 256}, {rgbint // 256 % 256}, {rgbint % 256})'
        
#         gt=(gt-gt.min())/(gt.max()-gt.min())
        
#         colors=bone_colormap
#         cmap= [[np.quantile(gt,i/(len(colors)-1)),rgb2string(colors[i])] for i in range(len(colors))]
#         return[[0,rgb2string(colors[0])],*cmap,[1,rgb2string(colors[-1])]]