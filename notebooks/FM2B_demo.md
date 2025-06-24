---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Fast marching to branching morphologies: Supplementary code
This is the accompanying code for the preprint at biorxiv: https://biorxiv.org/cgi/content/short/2024.11.16.623917v1
And manuscript submitted to Royal Society Interface journal

```{code-cell} ipython3
RunningInCOLAB = 'google.colab' in str(get_ipython())
RunningInCOLAB
```

```{code-cell} ipython3
if RunningInCOLAB:
  #! pip install scikit-fmm
  ! pip install git+https://github.com/scikit-fmm/scikit-fmm.git
  ! pip install scikit-image
  ! pip install powerlaw
```

```{code-cell} ipython3

```

```{code-cell} ipython3

if RunningInCOLAB:
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/balanced_mst.py
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/fm2b.py
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/morpho_trees.py
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/sca.py
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/speed_fields.py
  ! wget https://github.com/abrazhe/fm2b/raw/refs/heads/main/visualization.py
```

```{code-cell} ipython3
import os
import sys
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
import matplotlib as mpl
from matplotlib import pyplot as plt
```

```{code-cell} ipython3
#import cv2
```

```{code-cell} ipython3
from functools import reduce
import operator as op
```

```{code-cell} ipython3
from importlib import reload
```

```{code-cell} ipython3
import numpy as np
import scipy as sp
from scipy import ndimage as ndi

from pathlib import Path
```

```{code-cell} ipython3
from skimage.filters import sato
```

```{code-cell} ipython3

```

```{code-cell} ipython3
import powerlaw
```

```{code-cell} ipython3
from tqdm.auto import tqdm, trange
```

```{code-cell} ipython3
import pickle
```

```{code-cell} ipython3
import seaborn as sns
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
from numba import jit
```

```{code-cell} ipython3

```

```{code-cell} ipython3
if not Path('figures').exists():
    Path('figures').mkdir()
```

```{code-cell} ipython3
import skfmm
```

```{code-cell} ipython3
sys.path.append('../')
```

```{code-cell} ipython3
import morpho_trees as mt
import speed_fields as spf
import visualization as vis

import balanced_mst as bmst
import fm2b
```

```{code-cell} ipython3
def eu_dist(p1, p2):
    return np.sqrt(np.sum([(x - y)**2 for x, y in zip(p1, p2)]))

def cartesian2polar(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x,y)
    return r,theta

def polar2cartesian(r,theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y


def gen_random_polar_points(Npoints, rmin=0.1, rmax=1):
    radii = np.random.uniform(rmin,rmax, size=Npoints)
    theta = np.random.uniform(0, 2*np.pi, size=Npoints)
    return np.array([radii, theta]).T
```

```{code-cell} ipython3
def largest_region(mask):
    labels, nlab = ndi.label(mask)
    if nlab > 0:
        objs = ndi.find_objects(labels)
        sizes = [np.sum(labels[o]==k+1) for k,o in enumerate(objs)]
        k = np.argmax(sizes)
        return labels==k+1
    else:
        return mask
```

```{code-cell} ipython3
def flat_indices(shape):
    idx = np.indices(shape)
    return np.hstack([np.ravel(x_)[:,None] for x_ in idx])
```

```{code-cell} ipython3
def gauss2d(xmu=0, ymu=0, xsigma=10, ysigma=10):
    xsigma, ysigma = list(map(float, [xsigma, ysigma]))
    return lambda x,y: np.exp(-(x-xmu)**2/(2*xsigma**2) - (y-ymu)**2/(2*ysigma**2))

def gauss_blob(loc, sigma, shape):
    xx,yy = np.mgrid[:shape[0],:shape[1]]
    fn = gauss2d(xmu=loc[0],ymu=loc[1], xsigma=sigma,ysigma=sigma)
    return fn(xx,yy)
```

```{code-cell} ipython3
def sample_points(prob_map, size=500):
    locs = np.array(np.where(prob_map>0)).T
    idx = np.random.choice(len(locs), size=size, replace=False, p=prob_map[prob_map>0])
    return locs[idx]
```

```{code-cell} ipython3
def percentile_rescale(arr, plow=1, phigh=99):
    vmin,vmax = np.percentile(arr, (plow, phigh))
    if vmin == vmax:
        return np.zeros_like(arr)
    else:
        return np.clip((arr-vmin)/(vmax-vmin),0,1)


def clip_outliers(m, plow=0.5, phigh=99.5):
    px = np.percentile(m, (plow, phigh))
    return np.clip(m, *px)


@jit(nopython=True)
def local_jitter(v, sigma=5):
    L = len(v)
    vx = np.copy(v)
    Wvx = np.zeros(L)
    for i in range(L):
        j = i + int(np.round(np.random.randn() * sigma))
        j = max(0, min(j, L - 1))
        vx[i] = v[j]
        vx[j] = v[i]
    return vx
```

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
plt.rc('figure', dpi=150)
```

## Simple example fields and paths in them

```{code-cell} ipython3
sigmas = (1.5, 3, 6, 12)
```

```{code-cell} ipython3
field_constant = 0.5*np.ones((256,512))

field_gradient = field_constant*np.linspace(0,1,len(field_constant))[:,None]**0.5
field_random = np.sum([s**2*ndi.gaussian_filter(np.random.randn(256,512),s) 
                       for s in sigmas + (24,)],0)
field_random = 0.1 + percentile_rescale(field_random,0,100)
field_obstacle = field_constant.copy()
field_obstacle[50:,126:136] = 0
```

```{code-cell} ipython3
root =  (128, 500)
start =  (128, 12)
```

### The fields

```{code-cell} ipython3
example_fields = [field_constant, field_gradient, field_obstacle,field_random]
```

```{code-cell} ipython3
aspect = example_fields[0].shape[1]/example_fields[0].shape[0]
fig, axs = plt.subplots(2,2, figsize=(3*aspect,3), gridspec_kw=dict(hspace=0.05,wspace=0.05))
for field, ax in zip(example_fields, np.ravel(axs)):
    ax.imshow(field, vmin=0,vmax=1, cmap='gray')
    ax.plot(*root[::-1], 'r+', ms=10)
    ax.axis('off')
```

```{code-cell} ipython3
aspect
```

### Travel-times

```{code-cell} ipython3
phi0 = np.ones(example_fields[0].shape,dtype=bool)
phi0[root] = False
```

```{code-cell} ipython3
ttms = [skfmm.travel_time(phi0,speed=field) for field in example_fields]
```

```{code-cell} ipython3
fig, axs = plt.subplots(2,2, figsize=(3*aspect,3), gridspec_kw=dict(hspace=0.05,wspace=0.05))
for ttm, ax in zip(ttms, np.ravel(axs)):
    ax.imshow(percentile_rescale(ttm,1,99), cmap='rainbow_r')
    ax.plot(*root[::-1], 'k+', ms=10)
    ax.axis('off')
```

### GD Paths

```{code-cell} ipython3
trees = [fm2b.build_tree(ttm, [start],tm_mask=~phi0)[0] for ttm in ttms]
```

```{code-cell} ipython3
paths = [tree.tips[0].apath_to_root() for tree in trees]
```

```{code-cell} ipython3
fig, axs = plt.subplots(2,2, figsize=(3*aspect,3), gridspec_kw=dict(hspace=0.05,wspace=0.05))
for field, path, ax in zip(example_fields, paths, np.ravel(axs)):
    ax.imshow(field, vmin=0,vmax=1, cmap='gray')
    ax.plot(path[:,1],path[:,0], 'm')
    ax.plot(*root[::-1], 'r+', ms=10)
    ax.plot(*start[::-1], 'x', color='lime', ms=10)
    ax.axis('off')
```

```{code-cell} ipython3
seeds_extra = [(x, 12) for x in range(26,230,20)]
```

```{code-cell} ipython3
trees_extra = [fm2b.build_tree(ttm, seeds_extra,tm_mask=~phi0)[0]
               for ttm in ttms]
```

```{code-cell} ipython3
fig, axs = plt.subplots(2,2, figsize=(3*aspect,3), gridspec_kw=dict(hspace=0.05,wspace=0.05))
for field, tree, ax in zip(example_fields, trees_extra, np.ravel(axs)):
    ax.imshow(field, vmin=0,vmax=1, cmap='gray')
    vis.plot_tree(tree, random_colors=False, 
                   show_root=False, zorder=2,ax=ax)
    for start in seeds_extra:
        ax.plot(*start[::-1], 'x', color='lime', ms=5)
    ax.plot(*root[::-1], 'r+', ms=10)
    ax.axis('off')
```

## Merging GD paths in stochastic FMM-based travel-time maps

```{code-cell} ipython3
Npx = 512
X, Y = np.mgrid[:512,:512]
X.shape
```

```{code-cell} ipython3
field_shape = (Npx,Npx)
```

```{code-cell} ipython3
speed = spf.multiscale_Sato(field_shape)
phi0 = spf.central_phi0(field_shape)
ttx = skfmm.travel_time(phi0, speed=speed)
vis.show_tt_map(ttx, with_boundary=True)
```

```{code-cell} ipython3
init_pts_polar = gen_random_polar_points(20, 50, 225)
init_pts = np.array([polar2cartesian(*p) for p in init_pts_polar]) + (255,255)
```

```{code-cell} ipython3
init_pts_more = np.array([polar2cartesian(*p) 
                          for p in gen_random_polar_points(50, 50, 225)]) + (255,255)
```

```{code-cell} ipython3
vis.show_tt_map(ttx, with_boundary=False)
plt.plot(init_pts[:,0], init_pts[:,1], 'r.')
```

```{code-cell} ipython3
tree,fails = fm2b.build_tree(ttx, init_pts, tm_mask =~phi0)
```

```{code-cell} ipython3
len(tree),len(fails)
```

```{code-cell} ipython3
fig = plt.figure(figsize=(4,4))
vis.plot_tree(tree, random_colors=False, ax=plt.gca())
```

## Effect of speed field

```{code-cell} ipython3
# filaments_ms = percentile_rescale(sum(sato(np.random.randn(*field_shape), 
#                                            [s],black_ridges=False)
#                                       for s in (1.5, 3, 6, 12)))

satos = [sato(np.random.randn(*field_shape),[s],black_ridges=False)  
         for s in (1.5, 3, 6, 12)]


filaments_ms = np.mean([percentile_rescale(sc,0.1,99.9) for sc in satos],0)
#filaments_ms = np.mean([sc for sc in satos],0)

filaments_ms = percentile_rescale(filaments_ms,0,100)

filaments_hf = percentile_rescale(sato(np.random.randn(*field_shape), [1.5]))
filaments_lf = percentile_rescale(sato(np.random.randn(*field_shape), [6]))
```

```{code-cell} ipython3
plt.imshow(filaments_ms)
```

```{code-cell} ipython3
gauss_ms = percentile_rescale(sum(s**2*ndi.gaussian_filter(np.random.randn(*field_shape), s)
                                    for s in (1.5, 3, 6, 12)), 0, 100)
gauss_hf = percentile_rescale(ndi.gaussian_filter(np.random.randn(*field_shape), 1.5), 0, 100)
gauss_lf = percentile_rescale(ndi.gaussian_filter(np.random.randn(*field_shape), 6), 0, 100)
```

```{code-cell} ipython3
uniform_field = np.ones(field_shape)*0.85
uniform_field[0,0] = 0
```

```{code-cell} ipython3
fields = [
    uniform_field,
    gauss_hf,
    #gauss_lf,
    gauss_ms,
    filaments_hf*1.5,
    #filaments_lf,
    filaments_ms*1.5
]
```

```{code-cell} ipython3
# this is to ensure that the central point isn't isolated
for field in fields:
    field[~phi0] = np.median(field)
```

```{code-cell} ipython3
ttms = [skfmm.travel_time(phi0, speed=m) for m in fields]
```

```{code-cell} ipython3
fig,axs = plt.subplots(3,len(fields)+1, figsize=(9.1,5),
                       gridspec_kw=dict(width_ratios=[10]*len(fields)+[1])
                      )

inset_width = 32
inset_shift = 128
extent=(0,512,0,512)

morpho_acc = dict()

x1, x2, y1, y2 = (inset_shift, 
                  inset_shift+inset_width, 
                  inset_shift, 
                  inset_shift+inset_width, )

for ax,m in zip(axs[0],fields):
    spf_imh = ax.imshow(m, vmin=0,vmax=1, extent=extent, cmap='viridis')
    
    m_crop = m[512-inset_shift-inset_width:512-inset_shift,
              inset_shift:inset_shift+inset_width]
    
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33], 
                          xlim=(x1, x2), 
                          ylim=(y1, y2), 
                          xticks=[], yticks=[],
                         )
    axins.imshow(m,vmin=0,vmax=1,
                 extent=extent)
    
    ax.indicate_inset_zoom(axins, edgecolor="w")    
    for spine in axins.spines:
        axins.spines[spine].set_color('w')
    ax.axis('off')

    
for ax,tt in zip(axs[1],ttms):
    ttx = np.ma.filled(tt,np.nanmax(tt))
    ih = vis.show_tt_map(tt,ax, extent=extent,origin='lower',cmap='rainbow_r',interpolation='nearest')
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33], 
                      xlim=(x1, x2), 
                      ylim=(y1, y2), 
                      xticks=[], yticks=[],
                     )
    vmax = np.percentile(ttx[ttx<np.nanmax(ttx)],99)
    #vmax = np.percentile(tt[tt<np.nanmax(tt)],99)
    axins.imshow(ttx, vmin=0, vmax=vmax, cmap='rainbow_r', origin='lower', extent=extent)
    ax.indicate_inset_zoom(axins, edgecolor="w")


morpho_acc['tortuosity'] = list()
morpho_acc['root_angle'] = list()
morpho_acc['jitter'] = list()

for ax,tt in zip(axs[2],ttms,):
    tree, fails = fm2b.build_tree(tt, init_pts, tm_mask =~phi0)
    tree_more, fails = fm2b.build_tree(tt, init_pts_more, tm_mask =~phi0)

    tree_more_simple = tree_more.get_simple()

    mst = bmst.build_MSTree((255,255), init_pts,bf=0,progress_bar=False)
    Lmst = mst.total_wiring_length
    
    tree.add_morphometry(tree)
    tree_more.add_morphometry()
    tree_more_simple.add_morphometry()
    
    morpho_acc['tortuosity'].append([tree.get_tortuosity(tree)])
    morpho_acc['jitter'].append([tree.get_wriggliness()])
    morpho_acc['root_angle'].append([n.root_angle for n in tree.tips])
    
    if len(tree):
        vis.plot_tree(tree,ax,random_colors=False,
                       lw=0.75,linecolor='k')
        tort = np.mean(morpho_acc['tortuosity'][-1])
        wriggle = np.mean(morpho_acc['jitter'][-1])
        ax.set_title(f'tortuosity: {tort:1.2f},\n\
jitter: {wriggle:1.2f},\n\
excess wiring:{tree.total_wiring_length/Lmst:1.1f}',
                     horizontalalignment='left',
                     loc='left',
                     fontsize=9)
        ax.axis([0,512,512,0])
        ax.axis('off')

titles = ('uniform', 'high-pass Gauss', 'MS Gauss', 
          'high-pass filaments', 
          'MS filaments')

for ax,letter,title in zip(np.ravel(axs), 'abcde', titles):
    #ax.text(6,512-24, f'({letter})', backgroundcolor='w') 
    ax.set_title(title, fontsize=10)

cb1 = plt.colorbar(spf_imh, ax=axs[0,-1], 
                   ticks=[0,1],
                   fraction=0.9,
                   aspect=7,
                   shrink=0.5, 
                   label='speed')

cb2 = plt.colorbar(ih, ax=axs[1,-1], 
                   shrink=0.5,
                   aspect=7,
                   fraction=0.9,
                   ticks=[0,100*np.round(vmax/100)],
                   label='travel time',)                     

for ax in axs[:,-1]:
    ax.remove()


plt.tight_layout()
```

```{code-cell} ipython3
fig = plt.figure(figsize=(5,5))
ax = plt.gca()
tree,_ = fm2b.build_tree(tt, np.random.permutation(init_pts_more)[:20], tm_mask =~phi0)
vis.plot_tree(tree,random_colors=False,linecolor='gray',ax=ax,show_root=False,lw=0.85)
vis.plot_tree(tree.get_simple(),random_colors=False,linecolor='lime',ax=ax,lw=0.5,show_root=False)
plt.axis('equal')
plt.axis('off')
```

## Effect of speed field update

```{code-cell} ipython3
ttx,bmask = spf.make_ttmap_and_mask(filaments_ms, phi0)
```

```{code-cell} ipython3
vis.show_tt_map(ttx, with_boundary=True)
plt.imshow(bmask, alpha=0.25)
```

```{code-cell} ipython3
seeds_all = np.array(np.where(bmask)).T
seeds = np.random.permutation(seeds_all)[:500]
```

### Compare different nonlinearities

```{code-cell} ipython3
%%time 

Nseeds=250

seeds = np.random.permutation(seeds_all)[:Nseeds]

fig, axs = plt.subplots(3,4,  figsize=(8,5))


col = 0

uamps = [0, 1, 0.5, 2, 0.5]
algs = ['linear', 'log', 'linear', 'power', 'exp']

saved_tree=None

bp_acc = []

for col, (uam, alg) in enumerate(zip(uamps,algs[:-1])): 
    tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, phi0, seeds, 
                                                 speed_gamma=uam,
                                                 scaling=alg,
                                                 tm_mask=~phi0, batch_size=1)
    tree.add_morphometry()

    bp_acc.append(mt.get_branching_pattern(tree))
    
    if saved_tree is not None:
        vis.plot_tree(saved_tree, axs[0,col], 
                       random_colors=False, 
                       show_root=False,
                       linecolor='lime', lw=0.5)
    
    vis.plot_tree(tree, axs[0,col], 
                   random_colors=False, 
                   show_root=False,
                   linecolor='k', lw=0.75)


    
    axs[0,col].plot(255,255,marker='o',mfc='none',mec='violet',ms=10)
    axs[1,col].imshow(clip_outliers(np.log2(1+speed)), cmap='viridis')
    vis.show_tt_map(ttx, ax=axs[2,col],cmap='rainbow_r')
    axs[0,col].axis([0,512, 512,0])

    axs[0,col].set_rasterization_zorder(10000)
    
    if (uam >0) and saved_tree is None:
        saved_tree=tree

axs[0,0].set_ylabel('tree')
axs[0,1].set_ylabel('updated speed')
axs[0,2].set_ylabel('updated travel time')

#for ax,letter in zip(np.ravel(axs), 'abcd'):
#    ax.text(10,10, f'({letter})', backgroundcolor='w') 

for ax,letter in zip(np.ravel(axs), 
                     ('$f(x) = 0$', 
                      '$f(x)=\\log_2(1 + kx)$', 
                      '$f(x) = kx$',
                      '$f(x) = x^k$',
                      #'$f(x) = e^{kx}$'
                     )):
    ax.text(25,10, f'{letter}', backgroundcolor='w') 


for ax in np.ravel(axs):    
    ax.axis('off')

plt.tight_layout()
```

```{code-cell} ipython3
plt.figure(figsize=(2,2))
for rad,bp in bp_acc:
    plt.plot(np.linspace(0,1,len(bp)),ndi.gaussian_filter1d(bp,1.5))
vis.lean_axes(plt.gca())
plt.axhline(0, color='silver',lw=0.75,)
plt.xlabel('rel. distance to root')
```

### Test for negative $k$ in power-law dependence

```{code-cell} ipython3
Nseeds = 500
seeds = np.random.permutation(seeds_all)[:Nseeds]
```

```{code-cell} ipython3
tree_pos, speed_pos, ttx = fm2b.iterative_build_tree(filaments_ms, phi0, 
                                             seeds, 
                                             speed_gamma=1,
                                             scaling='power',
                                             tm_mask=~phi0, 
                                             batch_size=1)

bp_pos = mt.get_branching_pattern(tree_pos)
```

```{code-cell} ipython3
tree_neg, speed_neg, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                                 phi0, 
                                                 seeds, 
                                                 speed_gamma=-1,
                                                 scaling='power',
                                                 tm_mask=~phi0, 
                                                 batch_size=1)
tree_neg.add_morphometry()
bp_neg = mt.get_branching_pattern(tree_neg)
```

```{code-cell} ipython3
tree_flat, speed_flat, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                                 phi0, 
                                                 seeds, 
                                                 speed_gamma=0,
                                                 scaling='linear',
                                                 tm_mask=~phi0, 
                                                 batch_size=1)
bp_f = mt.get_branching_pattern(tree_flat)
```

```{code-cell} ipython3
tree_zero, speed_zero, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                                 phi0, 
                                                 seeds, 
                                                 speed_gamma=0,
                                                 scaling='power',
                                                 tm_mask=~phi0, 
                                                 batch_size=1)
tree_zero.add_morphometry()
bp_z = mt.get_branching_pattern(tree_zero)
```

```{code-cell} ipython3
fig,axes = plt.subplots(1,3, figsize=(9,3))

maps = (np.log2(1+m) for m in (speed_pos, speed_zero, speed_neg))
titles = ('$k = 1$', '$k = 0$', '$k = -1$')
for ax, m,t  in zip(axes, maps, titles):
    ax.imshow(clip_outliers(m))#, vmin=0, vmax=1)
    ax.axis('off')
    ax.set_title(t,fontsize=10)
plt.tight_layout()
```

```{code-cell} ipython3
#fig = plt.figure()
#ax = plt.gca()
fig, axs = plt.subplots(1,2, figsize=(9,3), 
                        gridspec_kw=dict(wspace=0.25, width_ratios=(1,1.5)))

ax = axs[0]
vis.plot_tree(tree_pos, show_root=True, lw=0.85, random_colors=False, linecolor='lime', ax=ax)
vis.plot_tree(tree_neg, show_root=True, lw=1, random_colors=False, linecolor='fuchsia', ax=ax)
vis.plot_tree(tree_zero, show_root=True, lw=0.85, random_colors=False, linecolor='k', ax=ax)
ax.axis('equal')
ax.axis('off')

ax = axs[1]
ax.plot(bp_pos[0]/bp_pos[0].max(),bp_pos[1], color='lime',lw=0.85, label='$k=1$')
ax.plot(bp_z[0]/bp_z[0].max(),bp_z[1], color='k',lw=0.85, label='$k=0$')
ax.plot(bp_neg[0]/bp_neg[0].max(),bp_neg[1], color='fuchsia',lw=1,label='$k=-1$')
ax.plot(bp_f[0]/bp_f[0].max(),bp_f[1], color='gray', ls='--',lw=1,label='flat')
vis.lean_axes(ax)
ax.legend()
ax.set_xlabel('rel. distance to root')
ax.set_ylabel('rel. #bifs $-$ #leaves')
```

```{code-cell} ipython3
mst = bmst.build_MSTree((255,255),seeds,bf=1,progress_bar=True)
```

```{code-cell} ipython3
Lmst = mst.total_wiring_length

L_pos = tree.total_wiring_length
L_zero = tree_zero.total_wiring_length
L_neg = tree_neg.total_wiring_length
L_flat = tree_flat.total_wiring_length
print(L_pos/Lmst, L_zero/Lmst, L_neg/Lmst, L_flat/Lmst)
```

```{code-cell} ipython3
[np.mean(tr.get_tortuosity()) for tr in (tree, tree_zero, tree_neg)]
```

## Comparison to MST (Cuntz)

### MST

```{code-cell} ipython3
bf_list = np.linspace(0, 0.9, 10)
Nseeds = 250
bf_list
```

```{code-cell} ipython3
acc = dict()
examples = dict()

for bf in tqdm(bf_list):
    acc[bf] = {'root_angle':[],
               'extra_wiring':[],
               'tortuosity':[],
               'wriggliness':[],
               'branching':[]}
    for i in (range(10)):
        seeds = np.random.permutation(seeds_all)[:Nseeds]
        tree = bmst.build_MSTree((255,255),seeds,bf=bf,progress_bar=False)
        mst = bmst.build_MSTree((255,255),seeds,bf=0,progress_bar=False)
        tree.add_morphometry()
        mst.add_morphometry()
        
        L = tree.total_wiring_length
        Lmst = mst.total_wiring_length
        
        rad,bp = mt.get_branching_pattern(tree)

        #bp = bp/(len(fm2b.get_bifurcations(tree)) + len(fm2b.get_tips(tree)))
        
        root_angles = [n.root_angle for n in tree.nodes]

        acc[bf]['root_angle'].append(np.nanmean(root_angles))
        acc[bf]['extra_wiring'].append(np.nanmean(L/Lmst))
        acc[bf]['branching'].append(bp)
        acc[bf]['wriggliness'].append(np.nanmean(tree.get_wriggliness()))
        acc[bf]['tortuosity'].append(np.nanmean(tree.get_tortuosity()))
    examples[bf] = tree
```

```{code-cell} ipython3
fig, axs = plt.subplots(1,5,sharex=True, figsize=(9,2))
#for ax, bf in zip(axs, [bf_list[0],bf_list[2],bf_list[4],bf_list[8], bf_list[-1]]):
for ax, bf in zip(axs, bf_list[::2]):
    example = examples[bf]
    vis.plot_tree(example,ax=ax,random_colors=False,linecolor='k',lw=0.5,show_root=False)
    ax.plot(255,255, 'o',color='violet',ms=10,mfc='none')
    ax.set_title(f'$bf={bf:1.1f}$',fontsize=10)
    ax.axis('off')
```

```{code-cell} ipython3
fig,axs = plt.subplots(1,4,figsize=(9,2.5),
                       gridspec_kw=dict(
                       width_ratios=(1,1,1.5,0.1),
                       wspace=0.75))

for bf in acc:
    print(bf)

    ax = axs[0]
    ax.plot(acc[bf]['extra_wiring'], 
            acc[bf]['tortuosity'], '.',
            color=plt.cm.coolwarm(bf),
            label=f'{bf:1.2f}')
    ax.set_ylabel('tortuosity')

    ax = axs[1]
    ax.plot(acc[bf]['extra_wiring'], 
            acc[bf]['root_angle'], '.',
            color=plt.cm.coolwarm(bf),
            label=f'{bf:1.2f}')
    ax.set_ylabel('root angle')

    avg_bp = ndi.gaussian_filter1d(np.mean(acc[bf]['branching'],0),1.5)
    rel_rad = np.linspace(0,1,len(avg_bp))
    axs[2].plot(rel_rad, avg_bp, color=plt.cm.coolwarm(bf))

for ax in axs[:2]:
    ax.set_xlabel('excess wiring')
    ax.axvline(1,color='silver',lw=0.5,zorder=-10)
    ax.set_xticks([1,1.5,2])
    ax.set_xlim(0.9,2.1)

axs[0].set(ylim = (0.85, 5.5))

# axs[1].set(ylim = (0.2, 1.5), yticks=np.pi/12*np.arange(1,7),
#            yticklabels=['π/12', 'π/6', 'π/4', 'π/3', '5π/12', 'π/2']
#           )


axs[1].set(ylim = (0.2, 1.5), yticks=np.deg2rad(np.arange(15, 90, 15)),
           yticklabels=[f'{x:1.0f}°' for x in np.arange(15, 90, 15)]
          )


axs[2].set(xlabel = 'rel. distance to root',
           ylabel = 'rel. $(N_{bif} - N_{tip})$',
           ylim = (-0.1, 0.2))


axs[2].axhline(0, color='silver',lw=0.5,zorder=-10)


h = axs[3].imshow(np.linspace(0,1).reshape(-1,1),cmap='coolwarm')
cb = plt.colorbar(h,ax=axs[3],fraction=0.33,aspect=15,label='$bf$')
axs[3].remove()

for ax in axs[:3]:
    vis.lean_axes(ax)
```

### Ranging update rate in FM2B

```{code-cell} ipython3
uamps = np.linspace(0,0.5,11)
```

```{code-cell} ipython3
uamps
```

```{code-cell} ipython3
acc2 = dict()
examples2 = dict()

for uamp in tqdm(uamps):
    acc2[uamp] = {'root_angle':[],
           'extra_wiring':[],
           'tortuosity':[],
           'wriggliness':[],
           'branching':[]}
    for i in range(3):
        speed_profile = spf.multiscale_Sato()
        phi0 = spf.central_phi0(speed_profile.shape)

        # prevent speed profile from being zero just in this point
        speed_profile[~phi0] = np.median(speed_profile)
        
        seeds = np.random.permutation(seeds_all)[:Nseeds]
        
        tree, _, _ = fm2b.iterative_build_tree(speed_profile, phi0, seeds, 
                                               speed_gamma=uamp,
                                               tm_mask=~phi0, batch_size=1,
                                               batch_size_alpha=1.1,
                                               progress_bar=False)
        simple = tree.get_simple()
        mst = bmst.build_MSTree((255,255),seeds,bf=0,progress_bar=False)
        tree.add_morphometry()
        simple.add_morphometry()
        mst.add_morphometry()
        
        L = tree.total_wiring_length
        Lmst = mst.total_wiring_length
        
        rad,bp = mt.get_branching_pattern(tree)

        #bp = bp/(len(fm2b.get_bifurcations(tree)) + len(fm2b.get_tips(tree)))
        
        #root_angles = [n.root_angle for n in tree.values()]
        root_angles = [n.root_angle for n in simple.values()]

        acc2[uamp]['root_angle'].append(np.nanmean(root_angles))
        acc2[uamp]['extra_wiring'].append(np.nanmean(L/Lmst))
        acc2[uamp]['branching'].append(bp)
        acc2[uamp]['wriggliness'].append(np.nanmean(tree.get_wriggliness()))
        acc2[uamp]['tortuosity'].append(np.nanmean(tree.get_tortuosity()))
    examples2[uamp] = tree        
```

```{code-cell} ipython3

```

```{code-cell} ipython3
fig,axs = plt.subplots(1,4,figsize=(9,2.5),
                       gridspec_kw=dict(
                       width_ratios=(1,1,1.5,0.1),
                       wspace=0.75))

for uamp in acc2:

    ax = axs[0]
    ax.plot(acc2[uamp]['extra_wiring'], 
            acc2[uamp]['tortuosity'], '.',
            color=plt.cm.Reds_r(uamp/np.max(uamps)),
            label=f'{uamp:1.2f}')
    ax.set_ylabel('tortuosity')

    ax = axs[1]
    ax.plot(acc2[uamp]['extra_wiring'], 
            acc2[uamp]['root_angle'], '.',
            color=plt.cm.Reds_r(uamp/np.max(uamps)),
            label=f'{uamp:1.2f}')
    ax.set_ylabel('root angle')

    avg_bp = ndi.gaussian_filter1d(np.mean(acc2[uamp]['branching'],0),1.5)
    rel_rad = np.linspace(0,1,len(avg_bp))
    axs[2].plot(rel_rad, avg_bp, color=plt.cm.Reds_r(uamp/np.max(uamps)))

for ax in axs[:2]:
    ax.set_xlabel('excess wiring')
    ax.axvline(1,color='silver',lw=0.5,zorder=-10)
    ax.set_xticks([1,1.5,2])
    ax.set_xlim(0.9,2.2)
    
    #ax.set_xticks([1,2,3,4])
axs[2].set_xlabel('rel. distance to root')
axs[2].set_ylabel('rel. $(N_{bif} - N_{tip})$')


axs[0].set(ylim = (0.85, 5.5))
#axs[1].set(ylim = (0.2, 1.3))

axs[1].set(ylim = (0.2, 1.5), yticks=np.deg2rad(np.arange(15, 90, 15)),
           yticklabels=[f'{x:1.0f}°' for x in np.arange(15, 90, 15)]
          )


axs[2].set(xlabel = 'rel. distance to root',
           ylabel = 'rel. $(N_{bif} - N_{tip})$',
           ylim = (-0.1, 0.2))

axs[2].axhline(0, color='silver',lw=0.5,zorder=-10)


h = axs[3].imshow(uamps.reshape(-1,1),cmap='Reds_r')
cb = plt.colorbar(h,ax=axs[3],fraction=0.33,aspect=15,label='update rate')
axs[3].remove()

for ax in axs[:3]:
    vis.lean_axes(ax)
```

```{code-cell} ipython3
fig, axs = plt.subplots(1,5,sharex=True, figsize=(9,2))



for ax, uamp in zip(axs, uamps[:-1:2][::-1]):
    example = examples2[uamp]
    vis.plot_tree(example,ax=ax,random_colors=False,linecolor='k',lw=0.5,show_root=False)
    ax.plot(255,255, 'o',color='violet',ms=10,mfc='none')
    ax.set_title(f'update rate: {uamp:1.1f}',fontsize=9)
    ax.axis('off')
```

### Ranging wavesource update in FM2B: closer to MST

```{code-cell} ipython3
len(seeds)
```

```{code-cell} ipython3
tree, _, _ = fm2b.iterative_build_tree(filaments_ms, phi0, seeds, 
                                       tm_mask=~phi0, 
                                       batch_size=1,
                                       batch_size_alpha=1,
                                       do_phi0_update=True,
                                       max_count_phi0=2,
                                       speed_gamma=0,
                                       progress_bar=False)
mst = bmst.build_MSTree((255,255),seeds,bf=0,progress_bar=False)
```

```{code-cell} ipython3
vis.plot_tree(tree)
```

```{code-cell} ipython3
nages = np.arange(2,250,50)
nages = np.array([2, 51, 101, 151, 251])
len(nages)
```

```{code-cell} ipython3
nages
```

```{code-cell} ipython3
acc3 = dict()
examples3 = dict()

for nage in tqdm(nages):
    acc3[nage] = {'root_angle':[],
           'extra_wiring':[],
           'tortuosity':[],
           'wriggliness':[],
           'branching':[]}
    for i in range(3):
        speed_profile = spf.multiscale_Sato(field_shape)

        # prevent speed profile from being zero just in this point
        speed_profile[~phi0] = np.median(speed_profile)
        
        seeds = np.random.permutation(seeds_all)[:Nseeds]
        
        tree, _, _ = fm2b.iterative_build_tree(speed_profile, phi0, seeds, 
                                               speed_gamma=0,
                                               do_phi0_update=True,
                                               max_count_phi0=nage,
                                               tm_mask=~phi0, 
                                               batch_size=1,
                                               batch_size_alpha=1.1,
                                               progress_bar=False)
        simple = tree.get_simple()
        mst = bmst.build_MSTree((255,255),seeds,bf=0,progress_bar=False)
        tree.add_morphometry()
        simple.add_morphometry()
        mst.add_morphometry()
        
        L = tree.total_wiring_length
        Lmst = mst.total_wiring_length
        
        rad,bp = mt.get_branching_pattern(tree)

        #bp = bp/(len(fm2b.get_bifurcations(tree)) + len(fm2b.get_tips(tree)))
        
        #root_angles = [n.root_angle for n in tree.values()]
        root_angles = [n.root_angle for n in simple.values()]

        acc3[nage]['root_angle'].append(np.nanmean(root_angles))
        acc3[nage]['extra_wiring'].append(np.nanmean(L/Lmst))
        acc3[nage]['branching'].append(bp)
        acc3[nage]['wriggliness'].append(np.nanmean(tree.get_wriggliness()))
        acc3[nage]['tortuosity'].append(np.nanmean(tree.get_tortuosity()))
    examples3[nage] = tree        
```

```{code-cell} ipython3
acc3.keys()
```

```{code-cell} ipython3
fig,axs = plt.subplots(1,4,figsize=(9,2.5),
                       gridspec_kw=dict(
                       width_ratios=(1,1,1.5,0.1),
                       wspace=0.75))

for uamp in acc2:
    ax = axs[0]
    ax.plot(acc2[uamp]['extra_wiring'], 
            acc2[uamp]['tortuosity'], '.',
            color=plt.cm.Reds_r(uamp/np.max(uamps)),
            label=f'{uamp:1.2f}')
    ax.set_ylabel('tortuosity')

    ax = axs[1]
    ax.plot(acc2[uamp]['extra_wiring'], 
            acc2[uamp]['root_angle'], '.',
            color=plt.cm.Reds_r(uamp/np.max(uamps)),
            label=f'{uamp:1.2f}')
    ax.set_ylabel('root angle, rad')
    avg_bp = ndi.gaussian_filter1d(np.mean(acc2[uamp]['branching'],0),1.5)
    rel_rad = np.linspace(0,1,len(avg_bp))
    axs[2].plot(rel_rad, avg_bp, color=plt.cm.Reds_r(uamp/np.max(uamps)))    

for nage in acc3:

    ax = axs[0]
    ax.plot(acc3[nage]['extra_wiring'], 
            acc3[nage]['tortuosity'], '+',
            color=plt.cm.Blues(nage/np.max(nages)),
            label=f'{nage:1.2f}')
    ax.set_ylabel('tortuosity')

    ax = axs[1]
    ax.plot(acc3[nage]['extra_wiring'], 
            acc3[nage]['root_angle'], '+',
            color=plt.cm.Blues(nage/np.max(nages)),
            label=f'{uamp:1.2f}')
    ax.set_ylabel('root angle')

    avg_bp = ndi.gaussian_filter1d(np.mean(acc3[nage]['branching'],0),1.5)
    rel_rad = np.linspace(0,1,len(avg_bp))
    axs[2].plot(rel_rad, avg_bp, color=plt.cm.Blues(nage/np.max(nages)))

for ax in axs[:2]:
    ax.set_xlabel('excess wiring')
    ax.axvline(1,color='silver',lw=0.5,zorder=-10)
    ax.set_xticks([1,1.5,2])
    ax.set_xlim(0.9,2.2)
    
    #ax.set_xticks([1,2,3,4])
axs[2].set_xlabel('rel. distance to root')
axs[2].set_ylabel('rel. $(N_{bif} - N_{tip})$')


axs[0].set(ylim = (0.85, 5.5))
#axs[1].set(ylim = (0.2, 1.5))

axs[1].set(ylim = (0.2, 1.5), yticks=np.deg2rad(np.arange(15, 90, 15)),
           yticklabels=[f'{x:1.0f}°' for x in np.arange(15, 90, 15)]
          )


axs[2].set(xlabel = 'rel. distance to root',
           ylabel = 'rel. $(N_{bif} - N_{tip})$',
           ylim = (-0.1, 0.2)
          )

axs[2].axhline(0, color='silver',lw=0.5,zorder=-10)


h = axs[3].imshow(nages.reshape(-1,1),cmap='Blues')
cb = plt.colorbar(h,ax=axs[3],fraction=0.33,aspect=15,label='leaf index')
axs[3].remove()

for ax in axs[:3]:
    vis.lean_axes(ax)
#plt.legend()
#plt.gca().set(xscale='log', yscale='log')
#vis.multi_savefig(fig, f'figures/fm2b-nage-examples-{Nseeds}_seeds')
```

```{code-cell} ipython3
#3*9/5
```

```{code-cell} ipython3
fig, axs = plt.subplots(3,5,sharex=True, figsize=(9,3*(9/5)))

for ax, bf in zip(axs[0], bf_list[::2]):
    example = examples[bf]
    vis.plot_tree(example,ax=ax,random_colors=False,linecolor='k',lw=0.5,show_root=False)
    ax.set_title(f'{bf:1.1f}',fontsize=9)

for ax, uamp in zip(axs[1], uamps[:-1:2][::-1]):
    example = examples2[uamp]
    vis.plot_tree(example,ax=ax,random_colors=False,linecolor='k',lw=0.5,
                   #rasterize=True,
                   show_root=False)
    ax.set_title(f'{uamp:1.1f}',fontsize=9)
    ax.set_rasterization_zorder(10000)


for ax, nage in zip(axs[2], nages):
    example = examples3[nage]
    vis.plot_tree(example,ax=ax,random_colors=False,linecolor='k',lw=0.5,
                   #rasterize=True,
                   show_root=False)
    ax.set_title(f'{nage-1}',fontsize=9)
    #ax.set_rasterized(True)
    ax.set_rasterization_zorder(10000)
    

for ax in np.ravel(axs):
    ax.plot(255,255, 'o',color='violet',ms=10,mfc='none')
    ax.axis('off')

vis.multi_savefig(fig, f'figures/fm2b-nage-example-trees-{Nseeds}_seeds-rasterized')    
```

## Figure: Sampling strategies

```{code-cell} ipython3
ttx,bmask = spf.make_ttmap_and_mask(filaments_ms, phi0)
bmask_filt =  largest_region(ndi.binary_fill_holes(np.ma.filled(bmask)))
```

```{code-cell} ipython3
reload(vis)
plasma_x = vis.make_seethrough_colormap()
jet_x = vis.make_seethrough_colormap('jet')
reds_x = vis.make_seethrough_colormap('Reds')
blues_x = vis.make_seethrough_colormap('Blues')
spectralr_x = vis.make_seethrough_colormap('Spectral_r')
cmxg = vis.make_seethrough_colormap('Greens_r')
```

```{code-cell} ipython3
central_prob = gauss_blob((255,255), 75, field_shape)*(bmask_filt)
central_prob = ndi.gaussian_filter(central_prob, 5)
central_prob = central_prob/np.sum(central_prob)
plt.imshow(central_prob)
```

```{code-cell} ipython3
periph_prob = gauss_blob((255,255), 200, field_shape) - gauss_blob((255,255), 175, field_shape)
periph_prob = periph_prob**2
periph_prob *= bmask_filt
periph_prob = ndi.gaussian_filter(periph_prob, 5)
periph_prob = periph_prob/np.sum(periph_prob)

plt.imshow(periph_prob)
```

```{code-cell} ipython3
uniform_prob = ndi.gaussian_filter(np.ones(field_shape)*bmask_filt,5)

uniform_prob = uniform_prob/np.sum(uniform_prob)
plt.imshow(uniform_prob)
```

```{code-cell} ipython3
uniform_locs = sample_points(uniform_prob) 
periph_locs = sample_points(periph_prob)
central_locs = sample_points(central_prob)

uniform_locs_s   = sorted(uniform_locs, key=lambda p: eu_dist(p, (255,255)))
```

```{code-cell} ipython3
periph_locs_dense = sample_points(periph_prob,2000)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
root_asym = (255, 400)
phi0_asym = np.zeros(field_shape,bool)
phi0_asym[root_asym] = True
phi0_asym = ~phi0_asym
```

```{code-cell} ipython3
speed_for_asym = filaments_ms*(2e-5 + 5*periph_prob)
plt.imshow(speed_for_asym)
```

```{code-cell} ipython3
uniform_locs_s2 = sorted(uniform_locs, 
                         #key=lambda x: -eu_dist(x, (0,255)))
                         key=lambda x: -eu_dist(x, (0,300)))
```

```{code-cell} ipython3
prob_vmax = np.max([np.max(uniform_prob), np.max(central_prob), np.max(periph_prob)])
```

```{code-cell} ipython3
len(uniform_locs_s)
```

```{code-cell} ipython3
tree_sampling_acc = dict()
```

```{code-cell} ipython3
fig, axs = plt.subplots(2,3, figsize=(9,6), gridspec_kw=dict(hspace=0.05, wspace=0.05))

cmap='GnBu'

# -- uniform prob, ordered sampling (center)
ax = axs[0,0]
tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s, 
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
tree_sampling_acc['uniform_center'] = tree
ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
vis.plot_tree(tree, random_colors=False,  linecolor='k', lw=0.75, 
               rasterized=False, ax=ax)
ax.text(10,10, '(a)')
ax.set_rasterization_zorder(0)


# -- uniform prob, ordered sampling (periphery)
ax = axs[0,1]
tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s[::-1], 
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
tree_sampling_acc['uniform_periphery'] = tree

ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25)
vis.plot_tree(tree, random_colors=False, linecolor='k', 
               rasterized=False, lw=0.75, ax=ax)
ax.set_rasterization_zorder(0)
ax.text(10,10, '(b)')

# -- taxis (uniform prob, ordered sampling, point)
ax = axs[0,2]
tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             uniform_locs_s2, 
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
tree_sampling_acc['uniform_up'] = tree
ax.imshow(np.ma.masked_less(uniform_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
vis.plot_tree(tree, random_colors=False,  linecolor='k', 
               rasterized=True, lw=0.75, ax=ax)
ax.text(10,10, '(c)')


# -- non-uniform prob (center), uniform sampling
ax = axs[1,0]
tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             central_locs, 
                                             scaling='linear',
                                             tm_mask=~phi0, batch_size=2)
ax.imshow(np.ma.masked_less(central_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
vis.plot_tree(tree, random_colors=False,  linecolor='k', 
               rasterized=False, lw=0.75,ax=ax)
ax.text(10,10, '(d)')
ax.set_rasterization_zorder(0)
tree_sampling_acc['center_random'] = tree


# -- non-uniform prob (periphery), uniform sampling
ax = axs[1,1]
tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             periph_locs, 
                                             scaling='linear',
                                             tm_mask=~phi0, 
                                             batch_size=2)
ax.imshow(np.ma.masked_less(periph_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )

vis.plot_tree(tree, random_colors=False, linecolor='k', 
               rasterized=True, 
               lw=0.75, ax=ax)
ax.text(10,10, '(e)')
tree_sampling_acc['periphery_random'] = tree

# -- fundus
ax = axs[1,2]
tree, speed, ttx = fm2b.iterative_build_tree(speed_for_asym, 
                                             phi0_asym, 
                                             periph_locs, 
                                             scaling='linear',
                                             tm_mask=~phi0_asym, 
                                             batch_size=2)
ax.imshow(np.ma.masked_less(periph_prob,1e-10), 
          vmax=prob_vmax,
          cmap=cmap, alpha=0.25, )
vis.plot_tree(tree, random_colors=False, linecolor='k', 
               rasterized=True, lw=0.75, ax=ax)
ax.text(10,10, '(f)')
tree_sampling_acc['fundus'] = tree


for ax in np.ravel(axs):
    ax.set_rasterization_zorder(10000)
    ax.axis('off')
fig.tight_layout()
#vis.multi_savefig(fig, 'figures/sampling-strategies-rastr2')
```

```{code-cell} ipython3
seed_variants = {'uniform_center': lambda : sorted(sample_points(uniform_prob),
                                                  key=lambda p: eu_dist(p, (255,255))),
                 'uniform_periphery':lambda : sorted(sample_points(uniform_prob),
                                                  key=lambda p: -eu_dist(p, (255,255))),
                 'uniform_up': lambda : sorted(sample_points(uniform_prob),
                                                  #key=lambda p: -eu_dist(p, (0,255))),
                                                   key=lambda p: -eu_dist(p, (0,300))),
                 'center_random':lambda : sample_points(central_prob),
                 'periphery_random':lambda :  sample_points(periph_prob),
                }
```

```{code-cell} ipython3
%%time

shared_kw = dict(phi0=phi0, 
                 tm_mask=~phi0,
                 batch_size=2,
                 progress_bar=False)

ntries = 3


tree_sampling_acc2 = dict()
support_trees2 = dict()

for key in seed_variants:
    tree_sampling_acc2[key] = []
    support_trees2[key] = []

for key,seedsx in seed_variants.items():
    for i in trange(ntries):

        speed_profile = spf.multiscale_Sato(field_shape)
        tree, _, _  = fm2b.iterative_build_tree(speed_profile, seeds = seedsx(), 
                                                **shared_kw)
        sup_tree = tree.get_simple()
        tree.add_morphometry()
        tree.count_occurences()
        
        sup_tree.add_morphometry()
        
        tree_sampling_acc2[key].append(tree)
        support_trees2[key].append(sup_tree)
```

```{code-cell} ipython3
support_trees2.keys()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
tree_sampl_sup = dict()
```

```{code-cell} ipython3
for key,tree in tree_sampling_acc.items():
    print(key)
    #fm2b.count_occurences(tree)
    tree_sup = tree.get_simple()
    tree_sampl_sup[key] = tree_sup
    tree_sup.add_morphometry(with_bif_angle=True)
    tree.add_morphometry()

    for n in tree.nodes:
        n.set_tropism_angle(np.array([1,0]))
    
    for key,n in tree_sup.items():
        n.set_tropism_angle(np.array([1,0]))

        if hasattr(n, 'bif_angle'):
            tree[key].bif_angle = n.bif_angle
```

```{code-cell} ipython3
for key,forest in tree_sampling_acc2.items():
    print(key)
    for tree,tree_sup in zip(forest, support_trees2[key]):
        tree_sup.add_morphometry(with_bif_angle=True)
        tree.add_morphometry()
    
        for n in tree.nodes:
            n.set_tropism_angle(np.array([1,0]))
        
        for n in tree_sup.nodes:
            n.set_tropism_angle(np.array([1,0]))
```

```{code-cell} ipython3
keys = ['uniform_center', 'uniform_periphery', 'uniform_up',
        'center_random', 'periphery_random']
```

```{code-cell} ipython3
root_angles = {key:
    [[n.root_angle for n in tree.nodes] for tree in forest]
              for key,forest in support_trees2.items()}

trop_angles = {key:
               [[n.trop_angle for n in tree.nodes] 
                 for tree in forest]
                for key,forest in support_trees2.items()
              }


bif_angles = {key:
    [[n.bif_angle for n in tree.nodes if hasattr(n, 'bif_angle')] for tree in forest]
              for key,forest in support_trees2.items()}


counts = {key:
          [[n.count  for n in tree.nodes if n.count>1] 
           for tree in forest] for key,forest in tree_sampling_acc2.items()}

logcounts = {key:
          [[np.log2(n.count)  for n in tree.nodes if n.count>1] 
           for tree in forest] for key,forest in tree_sampling_acc2.items()}

branching_patterns = {key:
                     [mt.get_branching_pattern(tree)[1]
                      for tree in forest]
              for key,forest in tree_sampling_acc2.items()}
```

```{code-cell} ipython3
keys = ['uniform_center', 'uniform_periphery', 'uniform_up',
        'center_random', 'periphery_random']
```

```{code-cell} ipython3
fig,axs = plt.subplots(1,4, figsize=(9,2), 
                       gridspec_kw=dict(wspace=0.75))


ax = axs[0]
sns.lineplot({k:np.mean(branching_patterns[k],0) for k in keys}, 
             ls='-',dashes=False,  palette='muted',ax=ax)
lh =ax.legend()
ax.set(xticks=[0,50,100],xticklabels=[0,0.5,1], 
       xlabel='rel. distance to root', 
       ylabel='rel. $N_{bif}-N_{tip}$')
lh.remove()

ax = axs[1]
#sns.kdeplot({k:np.concatenate(logcounts[k]) for k in keys}, cut=0, palette='muted',ax=ax)
#lh =ax.legend()
#lh.remove()

for key in keys:
    powerlaw.plot_ccdf(np.concatenate(counts[key]),ax=ax)


#ax.set_xlabel('$\\log_2$ leaf index')
ax.set(xlabel='leaf index',ylabel='exceedance')


ax = axs[2]
sns.kdeplot({k:np.concatenate(root_angles[k]) for k in keys}, cut=0, common_norm=False, palette='muted',ax=ax)
#sns.kdeplot({k:np.concatenate(bif_angles[k]) for k in keys}, cut=0, palette='muted',ax=ax)
lh =ax.legend()
lh.remove()
ax.set(xticks=[0,np.pi/2,np.pi], xticklabels=[0, 'π/2', 'π'])
ax.set_xlabel('root angle')

ax = axs[3]
sns.kdeplot({k:np.concatenate(trop_angles[k]) for k in keys}, cut=0, common_norm=False, palette='muted',ax=ax)
lh = ax.legend(['(a)','(b)','(c)','(d)','(e)'][::-1], reverse=True, fontsize=8,
               loc=(1.05,0))
ax.set(xticks=[0,np.pi/2,np.pi], xticklabels=[0, 'π/2', 'π'])
ax.set_xlabel('tropism angle')

for ax in axs[2:]:
    ax.set_ylabel('prob. density')

for ax in axs:
    vis.lean_axes(ax)    
```

## Figures 4-5. Dense seeds and branch diameters

```{code-cell} ipython3
print(np.sum(bmask_filt))
uniform_locs_dense = sample_points(uniform_prob, np.sum(bmask_filt)) 
```

```{code-cell} ipython3
len(uniform_locs_dense)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
Ntotal = len(uniform_locs_dense)
```

```{code-cell} ipython3
pts_fractions = [0.002, 0.008, 0.03, 0.13, 0.5]
pts_fractions
```

```{code-cell} ipython3
tree_kw = dict(scaling='linear', tm_mask=~phi0, batch_size=1, batch_size_alpha=1.1,)
                                            

params = [
    tree_kw,
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32),
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32),
    tree_kw | dict(do_phi0_update=True, max_count_phi0=32)]

pts_samplers = [
    lambda n: uniform_locs_dense[:round(int(n))],
    lambda n: local_jitter(uniform_locs_dense)[:round(int(n))],
    lambda n: local_jitter(np.array(sorted(uniform_locs_dense[:round(int(n))],
                                                       key=lambda x: -eu_dist(x, (255,255))))),
    lambda n: local_jitter(np.array(sorted(uniform_locs_dense[:round(int(n))],
                                                       key=lambda x: eu_dist(x, (255,255))))),
]
```

```{code-cell} ipython3
sample_trees = dict()
```

```{code-cell} ipython3
filaments_ms = spf.multiscale_Sato()
```

```{code-cell} ipython3
%%time 

fig1, axs = plt.subplots(len(params)+1,len(pts_fractions), figsize=(9,9),
                        sharex='col', sharey='col',
                        gridspec_kw=dict(hspace=0.05, wspace=0.05),
                       )

final_trees = []
labels = ['CR', 'VR', 'VP','VC']
colors= ["#005f73", "#0a9396", "#ca6702", "#ae2012"]


for j, (pset, sampler) in enumerate(zip(tqdm(params,desc='params'), pts_samplers)):
    tp_ratios = []
    for k,frac in enumerate(pts_fractions):
        n_pts = int(round(Ntotal*frac))
        pts = sampler(n_pts)
        
        if j==0:
            ax = axs[0,k]
            ax.plot(pts[:,0], pts[:,1], 'k,')
            ax.axis('square')
            ax.axis([0,512,512,0])
            ax.axis('off')
            title = f'{100*frac:1.1f}%' if frac<0.01 else f'{100*frac:1.0f}%'
            ax.set_title(title)

        ax = axs[j+1,k]
        if k == 0:
            ax.text(10,10, labels[j], color=colors[j], va='top')
        
        tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms,phi0,pts, **pset)
        ttxf = skfmm.travel_time(phi0, speed=speed)
        counts = tree.count_occurences(speed.shape)

        

        if frac < 0.005:
            gamma = 1.75
        elif gamma < 0.1:
            gamma = 2
        else:
            gamma = 2.25
        
        tree.assign_diameters(min_diam=0.25,gamma=gamma,max_diam=9)
        portrait = vis.make_portrait(tree, speed.shape, fill_soma=True, soma_mask=~phi0)
        top_p = 95 if frac < 0.01 else 99.5
        vmin,vmax=np.percentile(portrait[portrait>0],(1,top_p))
        #print(vmin,vmax)
        ax.imshow(portrait, vmin=0,vmax=vmax, cmap='BuPu')
        ax.plot(255,255, 'o', color='violet', mfc='none', ms=10)
        ax.axis('off')
        tip_source_ratio = len(tree.tips)/len(pts)
        tp_ratios.append(tip_source_ratio)
        print('---', j,frac,'tip/source ratio:',tip_source_ratio)
    #sample_trees[labels[j]] = tree
    final_trees.append((tree, ttxf, pts, tree.tips, tp_ratios))
plt.tight_layout()
vis.multi_savefig(fig1, 'figures/updated-phi0-patterns')
```

```{code-cell} ipython3
fig1
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
len(final_trees)
```

```{code-cell} ipython3
final_portraits = [vis.make_portrait(tree[0], filaments_ms.shape, 
                                 fill_soma=True, soma_mask=~phi0)
                   for tree in final_trees]
```

```{code-cell} ipython3
fig,axs = plt.subplots(2,2, figsize=(6,6))
for ax, p in zip(np.ravel(axs), final_portraits):
    ax.imshow(p, cmap='BuPu')
plt.tight_layout()
```

```{code-cell} ipython3

```

```{code-cell} ipython3
sample_trees = {lab:coll[0] for lab,coll in zip(labels, final_trees)}
support_trees = {lab:tree.get_simple() for lab,tree in sample_trees.items()}
```

```{code-cell} ipython3
for lab in sample_trees:
    sample_trees[lab].add_morphometry()
    support_trees[lab].add_morphometry()
```

```{code-cell} ipython3
#mst = bmst.build_MSTree((255,255),pts,bf=0.25)
```

```{code-cell} ipython3
labels, sample_trees.keys()
```

```{code-cell} ipython3
plt.figure(figsize=(2,2))
for label, color in zip(labels, colors):
    powerlaw.plot_ccdf([n.count for n in sample_trees[label].nodes 
                        if n.count>1],color=color)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
# plt.hist([np.log2(n.count) for n in sample_trees[labels[0]].values() if n.count>1],20);
# plt.hist([np.log2(n.count) for n in sample_trees[labels[1]].values() if n.count>1],20,alpha=0.5);
```

```{code-cell} ipython3
labels
```

```{code-cell} ipython3
#%time treex,twigs = fm2b.prune_tree_twigs(sample_trees[labels[0]])
%time btree = sample_trees[labels[2]].get_backbone(16)
```

```{code-cell} ipython3
%%time 

fig,axs = plt.subplots(2,3, figsize=(8,5),
                       #gridspec_kw=dict(hspace=0.1,wspace=0.1),
                       dpi=150,
                       sharex=True, sharey=True)
for ax, cut in zip(np.ravel(axs), tqdm((1,2,4,8,16,32))):
    for lab,color in zip(labels,colors):
        btree = sample_trees[lab].get_backbone(cut)
        r,bp = mt.get_branching_pattern(btree)
        rnorm = np.linspace(0,1,len(bp))
        ax.plot(rnorm, bp, color=color,label=lab)
        #ax.set_title(cut
        ax.text(0.1, 0.31, f'min leaf index: {cut}', weight='normal')
        #ax.text(100,0.9,f'min. {cut} leaves',fontsize=8)
        vis.lean_axes(ax)
lh = axs[1,-1].legend(loc=(1.01,0.1))
axs[0,0].set_ylabel('rel. $N_{bif}-N_{tip}$')
axs[1,0].set_xlabel('rel. distance to soma')
```

```{code-cell} ipython3
labels
```

```{code-cell} ipython3
%%time 

root_angles = {key:[n.root_angle for n in support_trees[key].tips] 
              for key in labels}

logcounts = {key:[np.log2(n.count) 
               for n in sample_trees[key].nodes if n.count>0] 
              for key in labels}

counts = {key:[n.count
               for n in sample_trees[key].nodes if n.count>1] 
              for key in labels}


tortuosities = {key:sample_trees[key].get_tortuosity() for key in labels}

jitters = {key:sample_trees[key].get_wriggliness() for key in labels}

branching_patterns = {key:
                      ndi.gaussian_filter1d(mt.get_branching_pattern(sample_trees[key])[1],3)
              for key in labels}
```

```{code-cell} ipython3

```

```{code-cell} ipython3
#root_angles['CR']
```

```{code-cell} ipython3
#ax = axs[2]
plt.figure(figsize=(2,2))
ax = plt.gca()
sns.kdeplot({k:root_angles[k] for k in labels}, cut=0,  palette=colors,ax=ax)
lh =ax.legend()
lh.remove()
ax.set(xticks=[0,np.pi/2,np.pi], xticklabels=[0, 'π/2', 'π'])
ax.set_xlabel('root angle')
```

```{code-cell} ipython3
labels
```

```{code-cell} ipython3
#plt.boxplot(jitters[labels[3]])
```

```{code-cell} ipython3
#data = {k:tortuosities[k] for k in labels}
```

```{code-cell} ipython3
plt.figure(figsize=(2,2))
ax = plt.gca()
sns.boxplot([tortuosities[k] for k in labels],
            width=0.5,
            ax=ax,fliersize=0)
lh =ax.legend()
lh.remove()
#ax.set_ylim(0,10)
ax.set(xticklabels=labels)
vis.lean_axes(ax)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
fig2 = plt.figure(figsize=(3,3))

#colors= ["#005f73", "#0a9396", "#ca6702", "#ae2012"]
#labels = ['CR', 'VR', 'VP','VC']

for lab,coll,color in zip(labels,final_trees,colors):
    tpr = coll[-1]
    plt.plot(pts_fractions, tpr, 'o-', label=lab,mfc='none',color=color)
plt.legend(loc=(1.05, 0.5), ncol=1)
ax = plt.gca()
ax.set(xscale='log', xlabel='seed density', ylabel='tip fraction')
vis.lean_axes(ax)
vis.multi_savefig(fig2, 'figures/updated-phi0-tip-fractions')
```

```{code-cell} ipython3
seeds = uniform_locs_dense[:int(round(Ntotal*0.03))]

extra_tree, speed, ttx = fm2b.iterative_build_tree(filaments_ms, 
                                             phi0, 
                                             seeds, 
                                             **(tree_kw | dict(do_phi0_update=True, max_count_phi0=32)))
                                             
plt.figure()
ax = plt.gca()
_ = extra_tree.count_occurences(speed.shape)
extra_tree.assign_diameters(min_diam=0.25,gamma=2.25,max_diam=9)
portrait = vis.make_portrait(extra_tree, speed.shape, fill_soma=True, soma_mask=~phi0)
plt.imshow(portrait)
plt.axis('off')
```

```{code-cell} ipython3
plt.figure(figsize=(3,3))
plt.imshow(clip_outliers(portrait**0.5), vmin=0,  cmap='BuPu')
ax = plt.gca()
tips = np.array([t.v for t in extra_tree.tips])

plt.plot(seeds[:,1], seeds[:,0], color='k', ls='', marker='.',ms=8,mfc='none')
plt.plot(tips[:,1], tips[:,0], 'r.',ms=4)
ax.axis('off')
plt.tight_layout()
plt.axis([250,350, 150,50])
```

```{code-cell} ipython3

```

```{code-cell} ipython3
fig3, axs = plt.subplots(3,4,
                        figsize=(10,3*10/4),
                        gridspec_kw=dict(wspace=0.6, 
                                         hspace=0.5,
                                         height_ratios=(1,1,0.85),
                                        ))

min_nodecount=16

for j,lab,coll,color in zip(range(100), labels,final_trees,colors):
    tree, ttxf, pts, tips, tp_ratios = coll
    atips = np.array([t.v for t in tips])
    acc = []
    for t in atips:
        p = tree[tuple(t)].apath_to_root()
        acc.append((len(p), ttxf[tuple(t)]))
    acc = np.array(acc)
    ax = axs[1,0]
    ax.hist(ttxf[*atips.T],50, log=False, color=color, 
            density=True, 
            alpha =0.5,
            label=f'_{lab}',
            lw=1.5);

    sns.kdeplot(ttxf[*atips.T],color=color,cut=0,ax=ax,label=lab)
    

    ax = axs[2,j]
    ax.text(10, 140, lab, color=color)
    ax.hexbin(acc[:,0], acc[:,1], cmap='Blues', bins='log', rasterized=True)
    ax.set(xlabel='path length, a.u.')
    ax.axis((0,650,-1,150))

    ax = axs[0,1]
    powerlaw.plot_ccdf(counts[lab],color=color,ax=ax)

    # Branching pattern
    ax = axs[0,0]
    btree = tree.get_backbone(min_nodecount)
    bp = mt.get_branching_pattern(btree)[1]
    ax.plot(np.linspace(0,1,len(bp)),
            ndi.gaussian_filter1d(bp,1.5),
            color=color)

#lh = axs[0,0].legend()


#
ax = axs[0,2]
sns.kdeplot({k:root_angles[k] for k in labels}, 
            cut=0, 
            common_norm=False,
            palette=colors,ax=ax)
lh =ax.legend()
lh.remove()
ax.set(xticks=[0,np.pi/2,np.pi], xticklabels=[0, 'π/2', 'π'])

ax = axs[0,3]
sns.boxplot([tortuosities[k] for k in labels], width=0.5,  palette=colors,  ax=ax,fliersize=0)
ax.set_ylim(0,10)
lh =ax.legend()
lh.remove()
ax.set(xticklabels=labels)

ax = axs[1,1]
for lab,coll,color in zip(labels,final_trees,colors):
    tpr = coll[-1]
    ax.plot(pts_fractions, tpr, 'o-', label=lab,mfc='none',color=color)
#ax.legend(loc=(1.05, 0.5), ncol=1,fontsize=6)
ax.set(xscale='log', xlabel='seed density', ylabel='tip fraction')

ax = axs[1,2]
ax.imshow(clip_outliers(portrait**0.5), vmin=0,  cmap='BuPu')
tips = np.array([t.v for t in extra_tree.tips])
ax.plot(seeds[:,1], seeds[:,0], color='k', ls='', marker='.',ms=4,mfc='none')
ax.plot(tips[:,1], tips[:,0], 'r.',ms=2)
ax.axis('off')
ax.axis([270,330, 140,60])


axs[0,0].set_xlabel('rel. distance to root')
axs[0,1].set_xlabel('leaf index')
axs[0,1].set_xlim(1,axs[0,1].get_xlim()[1])
axs[0,2].set_xlabel('root angle')

axs[1,0].set_xlabel('travel time, a.u.')

#axs[0,0].set_ylabel('prob. density')
axs[0,0].set_ylabel('rel. $N_{bif}-N_{tip}$')
axs[0,1].set_ylabel('exceedance')
axs[0,2].set_ylabel('prob. density')
axs[0,3].set_ylabel('tortuosity')


axs[1,0].set_ylabel('prob. density')
axs[1,0].set_xlim(-5,150)



for ax in np.ravel(axs):
    vis.lean_axes(ax)

axs[1,-1].remove()
```

## Multi-cellular network

```{code-cell} ipython3
def grid2points(X,Y):
    #return np.array(list(zip(np.ravel(X),np.ravel(Y))))
    return np.array([np.ravel(X),np.ravel(Y)]).T

def extract_edge_lengths(locs, tri):
    return np.array([np.sum((locs[edge[0]] - locs[edge[1]])**2)**0.5 for edge in tri.edges])
```

```{code-cell} ipython3
def dart_throwing(xrange, yrange, min_distance, max_points=None, niters=int(1e6)):
    pts = []
    if max_points is None:
        max_points = niters
    for i in trange(niters):
        new_point = np.random.uniform(*xrange), np.random.uniform(*yrange)
        if len(pts):
            neighbors = kdt.query_ball_point(new_point,min_distance)
            if not len(neighbors):
                pts.append(new_point)
        else:
            pts = [new_point]
        kdt = sp.spatial.KDTree(pts)
        if len(pts)>max_points:
            break
    return kdt.data
```

```{code-cell} ipython3
#scale_w = 2
scale_w = 1

field_shape_w = (scale_w*512, scale_w*256*3)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
seeds = dart_throwing((scale_w*10,scale_w*(512-10)), 
                      (scale_w*10,scale_w*(256*3-10)),
                      min_distance=scale_w*50,
                      niters=scale_w*5000)
```

```{code-cell} ipython3
field_shape_w
```

```{code-cell} ipython3
plt.plot(seeds[:,1],seeds[:,0], '.')
```

```{code-cell} ipython3
%%time 

filaments_ms_w = spf.multiscale_Sato(field_shape_w)
speed_w = filaments_ms_w
```

```{code-cell} ipython3
kdt = sp.spatial.KDTree(seeds)

px_locs = (np.indices(speed_w.shape)
           .reshape((2,-1))
           .T)

labels = kdt.query(px_locs)[1] + 1
labels = labels.reshape(speed_w.shape)



plt.imshow(labels, cmap=plt.cm.cividis)

plt.plot(seeds[:,1],seeds[:,0], 'm.')
plt.axis('off')
```

```{code-cell} ipython3
nn_dists = kdt.query(seeds,k=2)[0][:,1]
```

```{code-cell} ipython3
kde = sp.stats.gaussian_kde(nn_dists)
xfit = np.linspace(100,150)
```

```{code-cell} ipython3
plt.hist(nn_dists, 12, density=True, color='gray',histtype='stepfilled');
plt.plot(xfit,kde.evaluate(xfit))
mode = xfit[np.argmax(kde.evaluate(xfit))]
mode
```

```{code-cell} ipython3

```

```{code-cell} ipython3
valid_inits = kdt.query_ball_point(px_locs, 100,return_length=True)
```

```{code-cell} ipython3
somata_mask = kdt.query_ball_point(px_locs, scale_w*3,return_length=True).reshape(speed_w.shape)>0
```

```{code-cell} ipython3
len(valid_inits), len(px_locs)
```

```{code-cell} ipython3
#plt.imshow(somata_mask)
```

```{code-cell} ipython3
phi0_w = somata_mask
plt.imshow(phi0_w, interpolation='nearest')

phi0_w = ~phi0_w
```

```{code-cell} ipython3
plt.rc('figure', dpi=150)
```

```{code-cell} ipython3
ttx = skfmm.travel_time(phi0_w, speed=speed_w)
#ttx = ttx*(ttx>0)
#ttx[ttx.mask] = np.max(ttx)
#ttx = np.array(ttx)
ttx = np.ma.filled(ttx,np.max(ttx))

#boundary_mask = labels == 1
boundary_mask = ttx < np.percentile(ttx,90)

plt.figure(figsize=(9,6))
plt.imshow(ttx,cmap='rainbow_r', vmax=np.percentile(ttx[ttx<np.max(ttx)],99)); 
plt.colorbar(shrink=0.5)
#plt.contour(ttx, levels=[np.percentile(ttx, 25)], colors='c')
plt.contour(boundary_mask, levels=[0.5], colors='r',linewidths=0.75)
#plt.plot(seeds[:,1],seeds[:,0], '.', color=(0.2,0.2,0.2))
plt.axis('off')
```

```{code-cell} ipython3
#np.array(ttx)
```

```{code-cell} ipython3
#plt.imshow(boundary_mask)
```

```{code-cell} ipython3
np.percentile(ttx,0.1)
```

```{code-cell} ipython3
tm_mask_w = ndi.binary_dilation(ttx<=np.percentile(ttx,0.25*scale_w),iterations=1)
#tm_mask_w = ttx<=np.percentile(ttx,0.1)
plt.imshow(tm_mask_w)
```

```{code-cell} ipython3
uniform_prob_w = ndi.gaussian_filter(np.ones(field_shape_w)*boundary_mask,5)
uniform_prob_w /= uniform_prob_w.sum()
```

```{code-cell} ipython3
seeds_w = sample_points(uniform_prob_w, np.sum(uniform_prob_w>0)) 
```

```{code-cell} ipython3
len(seeds_w)*0.1
```

```{code-cell} ipython3

```

```{code-cell} ipython3
%%time

tree_w, speedx_w, ttx_w = fm2b.iterative_build_tree(speed_w, 
                                             phi0_w, 
                                             seeds_w[:int(len(seeds_w)*0.2)], 
                                             scaling='linear',
                                             tm_mask=tm_mask_w, 
                                             batch_size=1,
                                             batch_size_alpha=1.1)
plt.figure(); plt.imshow(np.log2(1+speedx_w), interpolation='nearest', cmap='PuBu')
counts = tree_w.count_occurences(speedx_w.shape)
plt.figure(); plt.imshow(np.log2(1+counts), interpolation='nearest', cmap='BuPu')

tree_w.assign_diameters(min_diam=0.25,gamma=2.25,max_diam=12)
portrait_w = vis.make_portrait(tree_w, speedx_w.shape, fill_soma=True, soma_mask=tm_mask_w)
plt.imshow(portrait_w,  cmap='BuPu')

plt.axis('off')
```

```{code-cell} ipython3
#vis.plot_tree(tree_w, random_colors=False,show_root=False,max_lw=1)
```

```{code-cell} ipython3
tree_w.add_morphometry();
```

```{code-cell} ipython3
somata_labels,nlab = ndi.label(tm_mask_w)
```

```{code-cell} ipython3
colors = np.random.rand(nlab+1,3)
colors[0] = (1,0,0)

palette = dict()

for root in tree_w.roots:
    loc = tuple(root.v)
    label = somata_labels[loc]
    palette[loc] = colors[label]

for pt in np.array(np.where(tm_mask_w)).T:
    pt = tuple(pt)
    palette[tuple(pt)] = colors[somata_labels[pt]]
```

```{code-cell} ipython3
#len(np.array(np.where(tm_mask_w)).T), np.sum(tm_mask_w)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
fig = plt.figure()

tree_w.assign_diameters(min_diam=0.25, gamma=1.8, max_diam=13)
portrait_w = vis.make_portrait(tree_w, speedx_w.shape, fill_soma=True, soma_mask=tm_mask_w)
plt.imshow(portrait_w,  cmap='BuPu')
plt.axis('off')
```

```{code-cell} ipython3
reload(vis)
```

```{code-cell} ipython3
%%time 
portrait_w2,colors = vis.make_portrait_colors(tree_w, speedx_w.shape, fill_soma=True, 
                                              palette=palette,
                                              soma_mask=tm_mask_w)
```

```{code-cell} ipython3
plt.imshow(1-np.clip(portrait_w2**0.5,0,1)[:,:,None]*colors,interpolation='nearest')
ax = plt.gca()
plt.tight_layout()
ax.axis('off')
```

```{code-cell} ipython3

```

```{code-cell} ipython3
len(seeds)
```

```{code-cell} ipython3
centerpoint = tuple(np.array(speed_w.shape)/2)
print(centerpoint)
kdt = sp.spatial.KDTree(seeds)
```

```{code-cell} ipython3
nearest = kdt.query(centerpoint)[1]
nearest_loc = tuple(seeds[nearest].astype(int))
```

```{code-cell} ipython3
selected_soma = [sm for k in range(1,somata_labels.max()+1) if (sm:=(somata_labels==k))[nearest_loc]][0]
```

```{code-cell} ipython3
ttx_fin = skfmm.travel_time(~selected_soma, portrait_w+0.01)
```

```{code-cell} ipython3
fig = plt.figure()
plt.imshow(portrait_w,  cmap='gray_r')

plt.axis('off')
tmax = 2718
h = plt.imshow(np.ma.masked_greater_equal(ttx_fin/1000,tmax/1000),vmax=tmax/1000, cmap='Spectral', alpha=0.5); 
plt.colorbar(h, ax=plt.gca(),shrink=0.25,aspect=15,label='travel time, a.u.')
plt.tight_layout()
```

```{code-cell} ipython3

```

## Test SCA

```{code-cell} ipython3
import sca
```

```{code-cell} ipython3
filaments_ms = spf.multiscale_Sato()
ttx, bmask =  spf.make_ttmap_and_mask(filaments_ms, phi0)
vis.show_tt_map(ttx, with_boundary=True)
plt.imshow(bmask, alpha=0.25)
seeds_all = np.array(np.where(bmask)).T
```

```{code-cell} ipython3
seeds = np.random.permutation(seeds_all)[:50]
tri = mpl.tri.Triangulation(*seeds[:,::-1].T)
edge_lengths = extract_edge_lengths(seeds, tri)


sns.histplot(edge_lengths, bins=25, color='gray', kde=True)
```

```{code-cell} ipython3
centerpoint = (255,255)
%time tree, missed = sca.build_SCATree(seeds,centerpoint,Dg=1,pq=100)
len(missed)
```

```{code-cell} ipython3
vis.plot_tree(tree,random_colors=False,linecolor='deepskyblue')
plt.plot(seeds[:,1], seeds[:,0], '.', ms=4, color='silver', zorder=-1)
#plt.plot(missed2[:,1], missed2[:,0], 'k+')
plt.plot(missed[:,1], missed[:,0], 'r.')
```

## Test MST

```{code-cell} ipython3
tree = bmst.build_MSTree((255,255),seeds,bf=0.2,progress_bar=False)
vis.plot_tree(tree)
```

## Execution times

```{code-cell} ipython3
import timeit
```

```{code-cell} ipython3
#num_points = np.arange(10,2000,20)
num_points = np.arange(10,2000//2,20)

ntries = 3
time_acc_mst = {}
for N in tqdm(num_points):
    time_acc_mst[N] = []
    for i in range(ntries):
        seeds = np.random.permutation(seeds_all)[:N]
        tick = timeit.timeit(lambda: bmst.build_MSTree((255,255),seeds,bf=0.2,progress_bar=False),number=1)
        time_acc_mst[N].append(tick)
```

```{code-cell} ipython3
time_acc_sca = {}
ntries=5
for N in tqdm(num_points):
    time_acc_sca[N] = []
    for i in range(ntries):
        seeds = np.random.permutation(seeds_all)[:N]
        tick = timeit.timeit(lambda: sca.build_SCATree(seeds,centerpoint,Dg=0.5), number=1)
        time_acc_sca[N].append(tick)
```

```{code-cell} ipython3
time_acc_fm2b = dict()
ntries=5
for N in tqdm(num_points):
    time_acc_fm2b[N] = []
    for i in range(ntries):
        seeds = np.random.permutation(seeds_all)[:N]
        tick = timeit.timeit(lambda : 
                             fm2b.iterative_build_tree(filaments_ms, phi0, seeds, 
                                                       tm_mask=~phi0, 
                                                       batch_size_alpha=1.1,
                                                       batch_size=1,
                                                       progress_bar=False
                                                      ), 
                             number=1)
        time_acc_fm2b[N].append(tick)
```

```{code-cell} ipython3
time_acc = dict()
ntries=5
for N in tqdm(num_points):
    time_acc[N] = []
    for i in range(ntries):
        seeds = np.random.permutation(seeds_all)[:N]
        tick = timeit.timeit(lambda : fm2b.build_tree(ttx, seeds, tm_mask =~phi0), number=1)
        time_acc[N].append(tick)
```

```{code-cell} ipython3
time_acc_Nseeds = {'MST':time_acc_mst,
                   'SCA':time_acc_sca,
                   'FM2B':time_acc_fm2b,
                   'FM2B, constant field':time_acc,
            }
```

```{code-cell} ipython3
avg_times_seeds = {key: [np.mean(acc[N]) for N in acc] for key,acc in time_acc_Nseeds.items()}
```

```{code-cell} ipython3
fig = plt.figure(figsize=(4,4))

for key, coll in avg_times_seeds.items():
    #print(coll)
    if len(coll):
        x = list(time_acc_Nseeds[key].keys())
        lh = plt.plot(x, coll, '.-', mfc='none', label=key)

ax = plt.gca()

ax.set(xlabel='number of seeds',
       ylabel='run time, s',
       xscale='log',
       yscale='log'
      )
plt.legend()
vis.lean_axes(ax)
```

```{code-cell} ipython3

```

## Effect of domain size

```{code-cell} ipython3
time_acc_size = {
                 'FM2B':dict(),
                 'FM2B, constant field':dict(),
                }
```

```{code-cell} ipython3
domain_sizes = [128, 256, 512, 1024, 2048, 3072]
Npts = 1000
ntries=3
```

```{code-cell} ipython3

```

```{code-cell} ipython3
acc = time_acc_size['FM2B, constant field']

for N in tqdm(domain_sizes):
    acc[N] = []
    for i in range(ntries):
        seeds_all = flat_indices((N,N))
        seeds = np.random.permutation(seeds_all)[:Npts]

        satos = [sato(np.random.randn(N,N),[s],black_ridges=False)  
                 for s in (1.5, 3, 6, 12)]
        filament_ms = np.mean([percentile_rescale(sc) for sc in satos],0)
        
        phi0 = np.zeros((N,N),dtype=bool)
        centerpoint = tuple(np.round(np.array([N,N])/2).astype(int))
        phi0[centerpoint] = True
        filament_ms[centerpoint] = np.median(filament_ms)
        phi0 = ~phi0

        ttx = skfmm.travel_time(phi0, speed=filament_ms)
        ttx = np.ma.filled(ttx, np.nanmax(ttx))
        
        tick = timeit.timeit(lambda : fm2b.build_tree(ttx, seeds, tm_mask =~phi0), number=1)
        acc[N].append(tick)
```

```{code-cell} ipython3
acc = time_acc_size['FM2B']
ntries=1
for N in tqdm(domain_sizes):
    acc[N] = []
    for i in range(ntries):
        seeds_all = flat_indices((N,N))
        seeds = np.random.permutation(seeds_all)[:Npts]

        satos = [sato(np.random.randn(N,N),[s],black_ridges=False)  
                 for s in (1.5, 3, 6, 12)]
        filament_ms = np.mean([percentile_rescale(sc) for sc in satos],0)
        
        phi0 = np.zeros((N,N),dtype=bool)
        centerpoint = tuple(np.round(np.array([N,N])/2).astype(int))
        phi0[centerpoint] = True
        filament_ms[centerpoint] = np.median(filament_ms)
        phi0 = ~phi0

        tick = timeit.timeit(lambda : 
                             fm2b.iterative_build_tree(filament_ms, phi0, seeds, 
                                                       tm_mask=~phi0, 
                                                       batch_size_alpha=1.1,
                                                       batch_size=1,
                                                       progress_bar=False
                                                      ), 
                             number=1)
        acc[N].append(tick)
```

```{code-cell} ipython3
avg_times_size = {key: [np.mean(acc[N]) for N in acc] for key,acc in time_acc_size.items()}
```

```{code-cell} ipython3
fig = plt.figure(figsize=(4,4))



for key, coll in avg_times_size.items():
    if len(coll):
        x = list(time_acc_size[key].keys())
        p1 = np.polyfit(x[:4],coll[:4], 1)
        p2 = np.polyfit(x[:4],coll[:4], 2)
        lh = plt.plot(x, coll, 's', mfc='none', label=key)
        xfit = np.linspace(x[0],x[-1])
        p = p1 if 'constant' in key else p2
        plt.plot(xfit, np.polyval(p, xfit), color=lh[0].get_color(), ls='--')

ax = plt.gca()

ax.set(xlabel='field side size (px)',
       xlim=(50,10000), 
       #ylim=(-1,250),
       ylabel='run time, s',
       xscale='log',
       yscale='log'
      )
plt.legend()
vis.lean_axes(ax)
```

```{code-cell} ipython3

```
