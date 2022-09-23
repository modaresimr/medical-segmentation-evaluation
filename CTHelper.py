import numpy as np
import cv2
def claheCT(img):
    imax=img.max()
    imin=img.min()
    normalimg=(img-imin)/(imax-imin)  
    
    clahe = cv2.createCLAHE(clipLimit=5.0)
    for i in range(img.shape[2]):
        img[:,:,i]=clahe.apply(np.uint8(normalimg[:,:,i]*255))

    return img/255 *(imax-imin)+imin

def normalizeImage(img):
    return (img-img.min())/(img.max()-img.min())  

def getROI(img,keepRatio=True):
    def get_contours(img):
        img = np.uint8(normalizeImage(img)*255)
        kernel = np.ones((3,3),np.float32)/9
        img = cv2.filter2D(img, -1, kernel)

        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)

        # filter contours that are too large or small
        ih, iw = img.shape
        totalArea = ih*iw
        contours2=contours
        for mx in np.arange(.5,1,.1):
            tmp = [cc for cc in contours if contourOK(cc, totalArea,mx)]
            if len(tmp)!=0:
                contours2=tmp
                break
        return contours2


    def contourOK(cc, totalArea,ignore_max=.9):
        x, y, w, h = cv2.boundingRect(cc)
        if ((w < 50 and h > 150) or (w > 150 and h < 50)) : 
            return False # too narrow or wide is bad
        area = cv2.contourArea(cc)
        # print(f'area={area}')
        return (area < totalArea * ignore_max) &(area> totalArea*.2)
    
    def find_boundaries(img, contours):
        # margin is the minimum distance from the edges of the image, as a fraction
        ih, iw = img.shape[0],img.shape[1]
        minx = iw
        miny = ih
        maxx = 0
        maxy = 0

        for cc in contours:
            x, y, w, h = cv2.boundingRect(cc)
            if x < minx: minx = x
            if y < miny: miny = y
            if x + w > maxx: maxx = x + w
            if y + h > maxy: maxy = y + h
        
        return (minx, miny, maxx, maxy)
    contours=[]
    for i in range(img.shape[2]):
        contours=[*contours,*get_contours(img[:,:,i])]
    bounds= find_boundaries(img, contours)
    roi=np.array([[bounds[1],bounds[3]],[bounds[0],bounds[2]]])
    if keepRatio:
        extend_roi_to_ratio(img.shape,roi,img.shape[1]/img.shape[0])
    idx = ()        
    for i in range(len(roi)):
        idx += (np.s_[roi[i][0]:roi[i][1] + 1],)
    # print(idx)
    return idx
        

def cropROI(img):
    return img[getROI(img)]



def extend_roi_shape(imgshape, roi, shape):
    for dim in range(len(shape)):
        if shape[dim]<=0:continue
        roi[dim][0]-=shape[dim]/2
        roi[dim][1]+=shape[dim]/2
        if roi[dim][0]<0:
            roi[dim][:]-=roi[dim][0]    
        elif roi[dim][1]>imgshape[dim]:
            roi[dim][:]-=roi[dim][1]-imgshape[dim]    
            
            
    return roi
    

def extend_roi_to_ratio(imgshape, roi,wh_ratio):
    w=roi[0][1] -roi[0][0]
    h=roi[1][1] -roi[1][0]
    
    nh=w*wh_ratio
    nw=h/wh_ratio
    # print(nw,w,nh,h)
    extend_roi_shape(imgshape,roi,[nw-w,nh-h])
    # if nw>w:
    #     extend(0,nw-w)
    # if nh>h:
    #     extend(1,nh-h)


def get_segment_roi(segments,margin=15,wh_ratio=1,mindim=[50,50,-1]):
    imgshape=segments[0].shape
    roi=np.zeros((len(imgshape),2),int)
    roi[:,0]=100000
        
    for seg in segments:
        nonzero = np.where(seg != 0)

        for i in range(len(nonzero)):
            if len(nonzero[i])>0:
                roi[i][0]=max(0,min(nonzero[i].min()-margin,roi[i][0]))
                roi[i][1]=min(imgshape[i],max(nonzero[i].max()+margin+ 1,roi[i][1]))
    
#     def extend(dim,siz):
#         tmp[dim][0]-=siz/2
#         tmp[dim][1]+=siz/2
#         if tmp[dim][0]<0:
#             tmp[dim][:]-=tmp[dim][0]
    
    for i in range(len(roi)):    
        if roi[i][1]<roi[i][0]:
            roi[i]=[0,imgshape[i]]
            
    extend_roi_shape(imgshape,roi,[mindim[i]- (roi[i][1]-roi[i][0]) for i in range(len(roi)) ])
    
    extend_roi_to_ratio(imgshape,roi,wh_ratio)
    
    # for i in range(len(tmp)):
    #     d=tmp[i][1] -tmp[i][0]
    #     if d<mindim[i]:
    #         diff=mindim[i]-d
    #         extend(i,diff)
            
    
#     w=tmp[0][1] -tmp[0][0]
#     h=tmp[1][1] -tmp[1][0]
#     nh=w*wh_ratio
#     nw=h/wh_ratio
#     # print(nw,w,nh,h)
#     if nw>w:
#         extend(0,nw-w)
#     if nh>h:
#         extend(1,nh-h)
    
    
    idx = ()        
    for i in range(len(roi)):
        idx += (np.s_[roi[i][0]:roi[i][1] + 1],)
    # print(idx)
    return idx

def upscale_ct(img,target_shape):
    newimg=np.zeros(target_shape)
    for i in range(img.shape[2]):
        newimg[:,:,i]=cv2.resize(img[:,:,i],(target_shape[1],target_shape[0]),interpolation=cv2.INTER_NEAREST)

    return newimg