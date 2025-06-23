
from tqdm import trange

import numpy as np
import scipy as sp

from morpho_trees import PathNode, Tree

def build_MSTree(p0, seeds, tree=None, bf=0.1, progress_bar=False):
    if tree is None:
        tree = Tree()
    
    loc0 = tuple(p0)
    
    if not loc0 in tree:
        root = PathNode(p0)
        root.path_length=0
        tree[loc0] = root
    else:
        root = tree[loc0]
    
    remaining = set(map(tuple, seeds))
    if loc0 in remaining:
        remaining.remove(loc0)
    remaining = remaining.difference(tree.keys())
    
    for i in trange(len(seeds), disable=not progress_bar):
        kdt_seeds = sp.spatial.KDTree(list(remaining))
        kdt_tree = sp.spatial.KDTree(list(tree.keys()))

        path_lengths = np.array([tree[m].path_length for m in tree])
        
        dm = (kdt_seeds
              .sparse_distance_matrix(kdt_tree, 1000,
                                      output_type='coo_matrix')
              .toarray()
             )
        
        dm2 = dm + bf*path_lengths
    
        chj,pj = np.unravel_index(dm2.argmin(), dm2.shape)

        child_loc = tuple(kdt_seeds.data[chj])
        parent_loc = tuple(kdt_tree.data[pj])
        dist = dm[chj,pj]
        
        parent = tree[parent_loc]
        child = PathNode(child_loc,parent=parent)
        child.path_length = parent.path_length + dist
        tree[child_loc] = child
        
        remaining = remaining.difference(tree.keys())
        if not len(remaining):
            break

    return tree