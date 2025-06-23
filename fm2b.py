import numpy as np
import itertools as itt
from functools import reduce, partial
import operator as op_        


from tqdm.auto import tqdm, trange

import scipy as sp

import skfmm

from morpho_trees import PathNode, Tree

# Note also
# (np.indices((3,3)) - 1).T.reshape(-1, 2)
# (np.indices((3,)*ndim) - 1).T.reshape(-1, ndim)


# Utility functions

def valid_loc(loc,shape):
    "Test if location not outside bounds"
    return reduce(op_.__and__, [(0 <= x < s) for x,s in zip(loc,shape)])

def neighbors(loc, shape):
    ndim = len(loc)
    shifts = set(itt.product([-1,0,1],repeat=ndim))
    shifts.discard((0,)*ndim)
    aloc = np.array(loc)
    locs = (aloc + shift for shift in shifts)
    return [tuple(loc) for loc in locs if valid_loc(loc, shape)]


# Building paths

def merging_rw_gd(field, p0, terminate_mask=None,  nsteps=10000, tree=None, 
                  success_at_stop = False,
                  pjitter=0.0):

    # tree is a hasmap, keys are locations, values are PathNodes
    if tree is None:
        tree = Tree()
    
    #if terminate_mask is None:
    #    terminate_mask = np.zeros(field.shape, bool)

    path_success = False
    
    p0 = tuple(map(int, p0))
    if p0 in tree:
        path_success = True
        nsteps = 0
        path = [tree[p0]]
    else:
        path = [PathNode(p0)]
        traj = [p0]

    visited = set(p0)

    for i in range(nsteps):
        prevnode = path[-1]
        p = tuple(prevnode.v)
        u = field[p]

        # look for nearest neighbors, not already visited 
        # when building the current path
        nns = (tuple(n) for n in np.array(neighbors(p, field.shape)).astype(int))
        nns = [n for n in nns if not n in visited]
        
        if not len(nns):
            # if there're no unvisited neighbors, stop
            if success_at_stop:
                path_success=True
            break
        nns = np.random.permutation(nns)
        nn_fields = np.array([field[tuple(n)] for n in nns])
        ksort = np.argsort(nn_fields)
        linked_nns = [tuple(n) for n in nns if tuple(n) in tree]
        linked_fields = [field[n] for n in linked_nns]

        # preferential attachment:
        if len(linked_nns):
            best = np.argmin(linked_fields)
            pnext = linked_nns[best]
        else:
            best = np.argmin(nn_fields)
            pnext = tuple(nns[best])
           
            if np.random.rand() < pjitter:
                # todo: approximately follow direction when choosing neighbor
                nns2 = (tuple(n) for n in 
                        np.array(neighbors(pnext, field.shape)).astype(int))
                nns = [pnext] + [n for n in nns2 if n in nns]
                pnext = nns[np.random.randint(len(nns))]
        
        # only create new node if this location hasn't been visited by other paths
        if pnext in tree:
            path_success = True
            node = tree[pnext]
            node.link(path[-1])
            break
        else:
            newnode = PathNode(pnext)
            visited.add(pnext)
            # now newnode is parent of prevnode
            newnode.link(prevnode)
            path.append(newnode)               
        
        if (terminate_mask is not None) and (terminate_mask[pnext]):
            path_success = True
            break
    if path_success:
        for p in path:
            tree[tuple(p.v)] = p
    else:
        for p in path:
            if p.parent:
                p.parent.unlink(p)
                p.parent=None
    return path, path_success


def build_tree(ttm, seeds, tree=None, tm_mask=None, success_at_stop=False, progress_bar=False):
    if tree is None:
        tree = Tree()
    ttm = np.ma.filled(ttm, np.max(ttm))
    fails = []
    for p in tqdm(seeds, disable=not progress_bar):
        # don't try to start from point already in the tree
        if tuple(p) in tree:
            continue
        path,success = merging_rw_gd(ttm, p, 
                                     terminate_mask=tm_mask, 
                                     pjitter=0.0, 
                                     nsteps=10000, tree=tree,
                                     success_at_stop=success_at_stop)
        if not success:
            fails.append(path)    
    return tree, fails



def iterative_build_tree(speed, phi0, seeds, 
                         update_amp=1,
                         tm_mask = None,
                         scaling='linear',
                         batch_size=1, 
                         batch_size_alpha=1,
                         speed_gamma = 2,
                         do_phi0_update=False,
                         max_count_phi0=1e20,
                         alpha=1.0,
                         progress_bar=True,
                         verbose=False,
                        ):
    speed0 = speed.copy()    
    speed = speed0.copy()
    
    ttx0 = skfmm.travel_time(phi0, speed=speed0)    
    ttx0 = np.ma.filled(ttx0, np.max(ttx0))
    ttx = ttx0.copy()
    
    tree = Tree()
    
    speed_upd = np.zeros(speed.shape) 
    occurence_count = np.zeros(speed.shape, 'uint16')
    count_unreachable =0
    count_seeds = 0
    fails = []

    ndim = np.ndim(speed)
    
    j = 0
    
    if scaling == 'linear':
        update_fn = lambda m:speed_gamma*m
    elif scaling == 'power':
        update_fn = lambda m:m**speed_gamma
    elif scaling == 'log':
        update_fn = lambda m: np.log2(1 + speed_gamma*m)
    elif scaling == 'exp':
        update_fn = lambda m: np.exp(m*speed_gamma)
    elif scaling == 'none':
        update_fn = lambda m: 0
    else:
        update_fn = lambda m:m
        
    
    for p0 in tqdm(seeds,disable=not progress_bar):
        count_seeds +=1
        p0 = tuple(map(int, p0))

        # skip unreacheable points
        if ttx[p0] == np.max(ttx):
            count_unreachable += 1
            continue
        
        
        try_path, finished = merging_rw_gd(ttx, p0, 
                                           terminate_mask=tm_mask, 
                                           pjitter=0.0, 
                                           nsteps=10000, 
                                           tree=tree)
        if finished:
            apath = tree[p0].apath_to_root()

            # update node occurences
            updated = tuple(apath[:-1,i] for i in range(ndim))
            occurence_count[updated] += 1

            # update speed field
            occ_mask = occurence_count > 0 # how to avoid memory allocations
            speed_upd[occ_mask] = update_fn(occurence_count[occ_mask]*1.0)
            
            #speed_upd[updated] += update_fn(update_amp)
            #speed += update_fn(speed_upd)

            speed = speed0 + speed_upd
            #return (apath, speed, speed_upd)
            if (not j%int(batch_size)) and alpha**j > 1e-6:    
                if do_phi0_update:
                    phi_try = ~((occurence_count < max_count_phi0)*occ_mask)
                    tm_mask = ~phi_try                
                else:
                    phi_try = phi0
                    
                ttx = skfmm.travel_time(phi_try, speed=speed)
                ttx = np.ma.filled(ttx, np.max(ttx))
                batch_size *= batch_size_alpha
            j += 1
        else:
            fails.append(np.array([n.v for n in try_path]))
            print('not finished for loc', p0) 
    if verbose:
        print('tree size:', len(tree))
        print('failed:', len(fails))
        print('visited:', count_seeds)
        print('unreachable points:', count_unreachable)
    return tree, speed, ttx


def build_MSTree(p0, seeds, tree=None, bf=0.1, progress_bar=True):
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

