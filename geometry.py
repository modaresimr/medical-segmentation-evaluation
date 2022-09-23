from scipy.ndimage import distance_transform_edt, generate_binary_structure, binary_erosion,binary_dilation

from auto_profiler import Profiler,Timer

def find_binary_boundary(binary_img,mode='thick',connectivity=1):
    """Return bool array where boundaries between labeled regions are True.
    Parameters
    ----------
    binary_img : array of int or bool
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:
        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.
    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    """
    result=np.zeros(binary_img.shape)
    binary_img=np.array(binary_img,bool)

    trimed_idx=myutils.array_trim(binary_img,margin=2,return_index=True)
    binary_img=binary_img[trimed_idx]
    
    ndim = binary_img.ndim
    footprint = generate_binary_structure(ndim, connectivity)    
    
    if mode=='inner':
        ero=binary_erosion(binary_img, footprint)
        boundaries=binary_img&(~ero)
    elif mode == 'outer':
        dil=binary_dilation(binary_img, footprint)
        boundaries = (~binary_img)&dil
    elif mode =='thick':
        dil=binary_dilation(binary_img, footprint)
        ero=binary_erosion(binary_img, footprint)
        boundaries = dil^ero
    else:
        raise Error(f'not supported mode {mode}')
    result[trimed_idx]=boundaries
    return result
    
        
        
        

        
        
import numpy as np
from scipy.ndimage import distance_transform_edt


def expand_labels(label_image, spacing=None):
    """Expand labels in label image by ``distance`` pixels without overlapping.

    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).

    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    spacing: iterable of floats, optional
        Spacing between voxels in each spatial dimension. If None, then the spacing between pixels/voxels in each dimension is assumed 1.

    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged

    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.  

    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.

    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.

    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`

    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])

    Labels will not overwrite each other:

    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])

    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.

    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """

    nearest_label_coords = distance_transform_edt(
        label_image == 0,return_distances=False, return_indices=True, sampling=spacing
    )
    
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    
    return nearest_labels




def skeletonize(img,spacing=None,surface=False):
    import skimage,myutils
    if spacing is None:
        spacing=[1,1,1]
    orig_img=img
    trimed_idx=myutils.array_trim(img,margin=2,return_index=True)
    img=img[trimed_idx]
    skel=np.zeros(orig_img.shape)
    spacing=np.array(spacing)
    spacing=spacing/spacing.min()
    
    img2s=skimage.transform.rescale(img,spacing,preserve_range=True,mode='edge')>0
    if surface:
        skel2=skimage.morphology.medial_surface(img2s)>0
    else:
        skel2=skimage.morphology.skeletonize_3d(img2s)>0
    skel[trimed_idx]=skimage.transform.resize(skel2,img.shape,preserve_range=True,mode='edge')>0
    
    return skel
