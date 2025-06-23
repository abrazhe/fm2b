import numpy as np

from skimage.filters import sato
from skimage.morphology import dilation, skeletonize, flood

from skimage import measure


import skfmm

def percentile_rescale(arr, plow=1, phigh=99):
    vmin,vmax = np.percentile(arr, (plow, phigh))
    if vmin == vmax:
        return np.zeros_like(arr)
    else:
        return np.clip((arr-vmin)/(vmax-vmin),0,1)
    

def multiscale_Sato(shape=(512,512), scales=(1.5, 3, 6, 12)):
    speed = percentile_rescale(
        #note `sato` from skimage already multiples by sigma^2
        sum(percentile_rescale(
            sato(np.random.randn(*shape), [s],black_ridges=False), 0.1, 99.9)
            for s in scales), 0, 100)
    return speed

def central_phi0(shape):
    center = tuple((np.array(shape)//2).astype(int))
    phi0 = np.zeros(shape, dtype=bool)
    phi0[*center] = True
    return ~phi0


def make_ttmap_and_mask(speed, phi0=None, percentile=50):
    if phi0 is None:
        phi0 = central_phi0(speed.shape)
    ttx = skfmm.travel_time(phi0, speed=speed)
    ttx = np.ma.filled(ttx, np.nanmax(ttx))
    bmask = ttx < np.percentile(ttx, percentile)
    return ttx, bmask