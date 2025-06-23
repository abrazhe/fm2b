import numpy as np

from numpy import array, arange, zeros, ones, sin, cos, pi
from numpy import linalg

from importlib import reload
import sys

from pathlib import Path

import itertools as itt

from matplotlib import pyplot as plt

import scipy as sp
from scipy import ndimage as ndi
from scipy import stats
import networkx

from tqdm.auto import tqdm


from morpho_trees import PathNode, Tree

class SCANode(PathNode):
    max_branches = 5
    def __init__(self, v, parent=None, tree=None):
        self.parent=parent
        self.children = []
        self.tree = Tree() if tree is None else tree
        loc = tuple(v)
        self.tree[loc] = self
        self.v = np.array(v)
        if parent is not None:
            parent.link(self)
        self.set_path_length()

    def spawn(self, 
              S : "attractor set", 
              Dg : "growth distance" = 0.025, 
              eps=0.00001,
              jitter=0.01,
              gamma=1,
              verbose=False):
        
        if not len(S):
            return
        S = np.array(S)
        d = (S - self.v)
        
        n = np.sum(d/(1e-6 + linalg.norm(d, axis=1)[:,np.newaxis]**gamma), axis=0)
                
        nnorm = np.linalg.norm(n)            
        
        n = n / (1e-6 + nnorm)
            
        vdash = self.v + Dg*n
                
        #if len(self.children) < self.max_branches:
        #    tip = TreeNode(vdash, parent=self, tree=self.tree)
        #    return True
        tip = None
        if len(self.children) < self.max_branches:
            tip = SCANode(vdash, parent=self, tree=self.tree)    
        return tip        

# # this should be converted to networkX Digraph eventually
# class TreeNode:
#     max_branches=5 # safety switch to prevent infinite branching
#     def __init__(self, v, parent=None, tree=None):
#         self.parent = parent
#         self.tree = set() if tree is None else tree
#         self.tree.add(self)
#         if parent is not None and not self in parent.children:
#             parent.children.append(self)
            
#         self.children = []
#         self.v = np.array(v) # spatial coordinate of the node
#         #self.dist_to_root = 0 if parent is None else parent.dist_to_root+1
    
#     def spawn(self, 
#               S : "attractor set", 
#               Dg : "growth distance" = 0.025, 
#               eps=0.00001,
#               jitter=0.01,
#               gamma=1,
#               verbose=False):
        
#         if not len(S):
#             return
#         S = np.array(S)
#         d = (S - self.v)
        
#         n = np.sum(d/(1e-6 + linalg.norm(d, axis=1)[:,np.newaxis]**gamma), axis=0)
                
#         nnorm = np.linalg.norm(n)            
        
#         n = n / (1e-6 + nnorm)
            
#         vdash = self.v + Dg*n
                
#         #if len(self.children) < self.max_branches:
#         #    tip = TreeNode(vdash, parent=self, tree=self.tree)
#         #    return True
#         tip = None
#         if len(self.children) < self.max_branches:
#             tip = TreeNode(vdash, parent=self, tree=self.tree)    
#         return tip


def space_colonization(tree, sources, 
                       iterations=20, 
                       Dg=0.025, Di=1, Dk=0.025, 
                       only_first_overextension=False,
                       progress_bar=False,
                       gamma=1,
                      ):

    in_empty_space = False
    used_overextension = False
        
    nodes_active = list(tree.nodes)#.copy()
    #print('1', tree_active)
    for j in tqdm(range(iterations),disable=not progress_bar):
        #print('2', tree_active)
        nodes_prev = [n for n in nodes_active if len(n.children) <= n.max_branches]
        
        if len(nodes_prev):
            kdt = sp.spatial.KDTree([n.v for n in nodes_prev])
        else:
            break
            
        
        d,inds = kdt.query(sources, distance_upper_bound=Di)
        #print(j, ':', len(d), np.min(d), Di)
            
        if (len(d) and np.min(d) > Di):
            in_empty_space = True
            #print(j, ':', len(d), np.min(d), 'in empty space')
            if  not used_overextension:
                d,inds = kdt.query(sources, distance_upper_bound=np.min(d))
                #print(np.min(d))
        else:
            if in_empty_space:
                in_empty_space=False
                if only_first_overextension:
                    used_overextension = True
        #if j > 50:
        #    used_overextension = True
                            
        spawned = []
        for i, n in enumerate(nodes_prev):
            S = sources[inds==i]
            new  = n.spawn(S, Dg, gamma)
            if new is not None:
                spawned.append(new)

        #print('3', spawned, tree_active)
        if not in_empty_space:
            nodes_active = [n for k,n in enumerate(nodes_prev) if k in inds] + spawned
        else:
            nodes_active = tree_prev + spawned
        
        too_close = kdt.query_ball_point(sources, Dk, return_length=True)        
        
        sources = sources[too_close == 0] 
        
        if (not len(sources)) or (not len(spawned)):
        #if not len(sources):
            break
        
        # add small jitter to break up ties
        sources  = sources + np.random.randn(*sources.shape)*Dg*0.05
    return tree, sources


from matplotlib.tri import Triangulation

def extract_edge_lengths(locs, tri):
    return np.array([np.sum((locs[edge[0]] - locs[edge[1]])**2)**0.5 for edge in tri.edges])


def build_SCATree(seeds, centerpoint, Dg=1, Dk=1, pq=100, di_mult=1):
    tree = Tree()
    root = SCANode(centerpoint, tree=tree)    
    tri = Triangulation(*seeds[:,::-1].T)
    edge_lengths = extract_edge_lengths(seeds, tri)
    
    Di  = di_mult*np.percentile(edge_lengths, pq) + 2*Dk + Dg
    #print(Di)
    
    new_tree, missed = space_colonization(tree, seeds, 
                                          iterations=500000,
                                          Di=Di,
                                          Dg=Dg,
                                          Dk=Dk,
                                          #max_dist_to_root=5000,
                                          progress_bar=False
                                          )
    return new_tree, missed

# def plot_tree(tree, sources=None, ax=None, random_colors=True, lw=1):
    
#     if ax is None:
#         fig, ax = plt.subplots(1,1)

#     color = np.random.rand(3) if random_colors else 'k'
#     for n in tree:
#         if n.parent is None:
#             ax.plot(n.v[0], n.v[1], 'ro')
#             #print('parent of node n is None')
#             #color = np.random.rand(3) if random_colors else 'k'
            
#         for ch in n.children:
#             vx = np.vstack([n.v, ch.v])
#             ax.plot(vx[:,0], vx[:,1], '-', lw=lw, alpha=0.95, color=color)
#     if sources is not None:
#         ax.plot(sources[:,0], sources[:,1], '.', color='gray', ms=0.5)
#     ax.axis('equal')


