#import graph_utils as gu
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl

import scipy as sp
from tqdm.auto import tqdm
from scipy import ndimage as ndi

def eu_dist(p1, p2):
    return np.sqrt(np.sum([(x - y)**2 for x, y in zip(p1, p2)]))

def multi_savefig(fig, name, formats =('svg', 'png'), **kwargs):
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'
    for f in formats:
        fig.savefig('.'.join([name,f]), **kwargs)


def lean_axes(ax, is_twin=False,
              dropped=True, hide = ('top', 'right')):
    """plot only x and y axis, not a frame for subplot ax"""

    if dropped:
        for key in ('top', 'right','bottom', 'left'):
            ax.spines[key].set_position(('outward', 6))

    for key in hide:
        ax.spines[key].set_visible(False)

    ax.get_xaxis().tick_bottom()
    if not is_twin:
        ax.get_yaxis().tick_left()
    else:
        ax.get_yaxis().tick_right()
    return 


def make_seethrough_colormap(base_name='plasma', gamma=1.5,kcut=5):
    cm_base = plt.cm.get_cmap(base_name)
    cmx = cm_base(np.linspace(0,1,256))
    v = np.linspace(0,1,256)
    cmx[:,-1] = np.clip(2*(v/(0.15 + v))**gamma,0,1)
    cmx[:kcut,-1] = 0
    cdict = dict(red=np.array((np.arange(256)/255, cmx[:,0], cmx[:,0])).T,
                 green=np.array((np.arange(256)/255, cmx[:,1], cmx[:,1])).T,
                 blue= np.array((np.arange(256)/255, cmx[:,2], cmx[:,2])).T,
                 alpha= np.array((np.arange(256)/255, cmx[:,3], cmx[:,3])).T)
    cm = mpl.colors.LinearSegmentedColormap(base_name + '-x',  cdict, 256)
    return cm


def make_portrait(tree, shape, min_diam_show=0, fill_soma=False, 
                  scale=1,
                  soma_mask=None,verbose=False):
    if soma_mask is None:
        soma_mask = np.zeros(shape, bool)
    portrait = np.zeros(shape)
    ndim = len(shape)
    px_locs = np.indices(shape).reshape((ndim,-1)).T
    ktree = sp.spatial.KDTree(px_locs)
    for loc, n in tqdm(tree.items(),disable=not verbose):
        diam = scale*n.diam
        xloc = [int(round(scale*coord)) for coord in loc]
        if diam >= min_diam_show:
            amp = np.log10(0.1+n.count)
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            knns = ktree.query_ball_point(xloc, diam/2)
            locs = px_locs[knns]
            for loc_ in locs:
                dist = eu_dist(xloc, loc_)
                ampr = amp*np.exp(-dist**2/(diam**2/4))
                l = tuple(loc_)
                portrait[l] = np.maximum(portrait[l],ampr)
    if fill_soma:
        portrait[soma_mask] = np.percentile(portrait[ndi.binary_dilation(soma_mask)],99)
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait


def make_portrait_colors(tree, shape, min_diam_show=0, fill_soma=False, 
                         palette=None,
                         soma_mask=None,verbose=False):
    if soma_mask is None:
        soma_mask = np.zeros(shape, bool)
    portrait = np.zeros(shape)
    colors = np.zeros(shape+(3,))
    ndim = len(shape)
    px_locs = np.indices(shape).reshape((ndim,-1)).T
    ktree = sp.spatial.KDTree(px_locs)
    for loc, n in tqdm(tree.items(),disable=not verbose):
        diam = n.diam
        if diam >= min_diam_show:
            amp = np.log10(0.1+n.count)
            color = palette[tuple(n.root_v)]
            #amp = diam
            #portrait += amp*gauss_blob(n, diam/2, portrait.shape)
            knns = ktree.query_ball_point(loc, diam/2)
            locs = px_locs[knns]
            for loc_ in locs:
                dist = eu_dist(loc, loc_)
                ampr = amp*np.exp(-dist**2/(diam**2/4))
                l = tuple(loc_)
                portrait[l] = np.maximum(portrait[l],ampr)
                colors[l] = color
    if fill_soma:
        portrait[soma_mask] = np.percentile(portrait[ndi.binary_dilation(soma_mask)],99)
        for pt in np.array(np.where(soma_mask)).T:
            loc = tuple(pt)
            if loc in palette:
                colors[loc] = palette[loc]
    #portrait = np.maximum(portrait, np.max(portrait)*gauss_blob((255,255), 10, counts.shape))
    return portrait, colors
    


def plot_tree(tree, ax=None, random_colors=True, linecolor='m', 
              show_root=True,
              lw=1, max_lw=10,
              rasterized=False,
              zorder=-1
             ):
    
    if ax is None:
        fig, ax = plt.subplots(1,1)

    color = np.random.rand(3) if random_colors else linecolor
    for loc,n in tree.items():
        if n.parent is None and show_root:
            #ax.plot(n.v[1], n.v[0], 'r.')
            ax.plot(n.v[1], n.v[0], 'o', color='violet', mfc='none', ms=10)
        
        for ch in n.children:
            vx = np.vstack([n.v, ch.v])
            if hasattr(ch,'diam'):
                lw = min(max_lw, ch.diam)
            else:
                lw = lw
                
            ax.plot(vx[:,1], vx[:,0], '-', lw=lw, alpha=0.95, 
                    rasterized=rasterized,
                    zorder=zorder,
                    color=color)
    ax.axis('equal')
    

def show_tt_map(ttm, ax=None, with_cbar=False, with_boundary=False,
                boundary_percentile=50,
                cmap='BuPu',
                **imshow_kws):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ttx = np.ma.filled(ttm,np.nanmax(ttm))
    #ttx = ttm
    vmax = np.percentile(ttx[ttx<np.max(ttx)],99)
    ih = ax.imshow(ttx, vmin=0, vmax=vmax, cmap=cmap,**imshow_kws)
    if with_cbar:
        plt.colorbar(ih, ax=ax)
    if with_boundary:
        boundary_mask = ttx < np.percentile(ttx,boundary_percentile)
        boundary_mask = ndi.binary_fill_holes(boundary_mask)
        plt.contour(boundary_mask, levels=[0.5], colors='r',linewidths=0.75)
    ax.axis('off')
    return ih
    

# def view_graph_as_shapes(g, viewer, color=None, kind='points', name=None):
#     """
#     display nodes of graph g in napari viewer as points or as lines
#     """
#     if color is None:
#         color = np.random.rand(3)
#     pts = np.array(g.nodes)

#     kw = dict(face_color=color, 
#               edge_color=color, 
#               blending='translucent_no_depth', 
#               name=name)
#     #kw = dict(face_color=color, edge_color=color,  name=name)
#     if kind == 'points':
#         viewer.add_points(pts, size=1, symbol='square', **kw)
#     elif kind == 'path':
#         viewer.add_shapes(pts, edge_width=0.5, shape_type='path', **kw)

# def view_graph_as_colored_image(g,  shape,
#                                 viewer=None, name=None,
#                                 root_chooser=lambda r: True,
#                                 scale=None,
#                                 change_color_at_branchpoints=False):
#     """
#     Convert a graph to a colored 3D stack image and add it to a napari viewer.
#     if the viewer instance is None, just return the colored 3D stack
#     """
#     paths = graph_to_paths(g, root_chooser=root_chooser)
#     stack = paths_to_colored_stack(paths, shape, change_color_at_branchpoints)
#     if viewer is not None:
#         viewer.add_image(stack, channel_axis=3, colormap=['red','green','blue'], name=name, scale=scale)
#         return viewer
#     else:
#         return stack

# def graph_to_paths(g, min_path_length=1, root_chooser=lambda r:True):
#     """
#     given a directed graph, return a list of a lists of nodes, collected
#     as unbranched segments of the graph
#     """

#     roots = gu.get_roots(g)

#     def _acc_segment(root, segm, accx):
#         if segm is None:
#             segm = []
#         if accx is None:
#             accx = []
#         children = list(g.successors(root))

#         if len(children) < 1:
#             accx.append(segm)
#             return

#         elif len(children) == 1:
#             c = children[0]
#             segm.append(c)
#             _acc_segment(c, segm, accx)

#         if len(children) > 1:
#             #segm.append(root)
#             accx.append(segm)
#             for c in children:
#                 _acc_segment(c, [root, c], accx)

#     acc = {}
#     for root in roots:
#         if root_chooser(root):
#             px = []
#             _acc_segment(root, [], px)
#             acc[root] = [s for s in px if len(s) >= min_path_length]
#     return acc


# def paths_to_colored_stack(paths, shape, change_color_at_branchpoints=False):
#     #colors = np.random.randint(0,255,size=(len(paths),3))
#     stack = np.zeros(shape + (3,), np.uint8)
#     for root in paths:
#         color =  np.random.randint(0,255, size=3)
#         for kc,pc in enumerate(paths[root]):
#             if change_color_at_branchpoints:
#                 color = np.random.randint(0,255, size=3)
#             for k,p in enumerate(pc):
#                 #print(k, p)
#                 p  = np.round(p).astype(int)
#                 # todo add interpolation?
#                 stack[tuple(p)] = color
#     return stack
