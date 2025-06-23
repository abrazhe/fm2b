
import numpy as np

from functools import reduce
import operator as op_        

from collections import deque


from tqdm.auto import tqdm


import pandas as pd

def eu_dist(p1, p2):
    return np.sqrt(np.sum([(x - y)**2 for x, y in zip(p1, p2)]))


def vec_angle(v1,v2):
    v1n = np.linalg.norm(v1)
    v2n = np.linalg.norm(v2)
    return np.arccos( np.dot(v1,v2)/(v1n*v2n))


# --  Class definitions

class PathNode:
    def __init__(self, loc, parent=None):
        self.v = np.array(loc)
        self.children = []
        self.parent = parent
        self.count = 0
        if parent is not None:
            parent.link(self)
            
    def link(self, child):
        child.parent = self
        if not child in self.children:
            self.children.append(child)

    def unlink(self,child):
        child.parent = None
        if child in self.children:
            self.children = [ch for ch in self.children if ch !=child]
            
    def __repr__(self):
        strx = 'Path node @({:1.2f}, {:1.2f})\n with {} successors'.format(*self.v,len(self.children))
        return strx

    def follow_to_root(self, max_nodes=1000000):
        acc = [self]
        tip = self
        for i in range(max_nodes):
            parent = tip.parent
            if parent is None:
                break
            tip = parent
            acc.append(tip)
            if i >= max_nodes-1:
                print('limit reached')
                break
        return acc
    
    def apath_to_root(tip):
        return np.array([n.v for n in tip.follow_to_root()])


    def set_path_length(self):
        if hasattr(self, 'path_length'):
            return self.path_length
            
        p = self.parent
        if p is None:
            self.path_length=0
            return 0
        elif hasattr(p, 'path_length') and p.path_length is not None:
            dist = eu_dist(self.v, p.v)
            self.path_length=dist + p.path_length
            return self.path_length
        else:
            print("error node parent is not None\
            and doesn't have `path_length` attribute")
            return None


    def set_root_loc(self):
        if hasattr(self,'root_v'):
            return
        p = self.parent
        if p is None:
            self.root_v = self.v
        else:
            self.root_v = p.root_v

    def set_tortuosity(self):
        self.set_root_loc()
        self.set_path_length()
        p = self.parent
        if p is None:
            self.tortuosity = 1
        else:
            self.tortuosity = self.path_length/(1e-6+eu_dist(self.v,self.root_v))
        return self.tortuosity


    def set_root_angle(self):
        self.set_root_loc()
        p = self.parent
        if p is None:
            self.root_angle=0
        else:
            v1 = p.v - self.v
            v2 = self.root_v - self.v
            theta = vec_angle(v1,v2)
            self.root_angle=theta
        return self.root_angle

    def set_bifurcation_angle(self):
        p = self.parent
        
        if p is None or (len(self.children)<2):
            self.bif_angle=0
            return
        
        child_counts = [ch.count for ch in self.children]
        
        child = self.children[np.argmin(child_counts)]

        tip = child.dfs_subtree(fn=lambda n:n)[-1]
        
        angle = np.pi-vec_angle(p.v-self.v,tip.v-self.v)
        self.bif_angle=angle

    def set_tropism_angle(self,trop_vector=None):
        if trop_vector is None:
            trop_vector = np.array([0,1])
        p = self.parent
        
        if p is None:
            self.trop_angle = 0
            
        else:
            v1 = p.v - self.v
            theta = vec_angle(v1,trop_vector)
            self.trop_angle = theta
        return self.trop_angle

    def dfs_subtree(self, fn=lambda n:None, max_iters=int(1e9)):
        stack = deque()
        discovered=set()
        stack.append(self)
        acc = []
        for i in range(max_iters):
            if not len(stack):
                break
            n = stack.pop()
            if not n in discovered:
                out = fn(n)
                acc.append(out)
                discovered.add(n)
                for ch in n.children:
                    if not ch in discovered:
                        stack.append(ch)
        return acc
    
    def prune_twig(self, min_length=10, max_count_diff=10):
        tip = self
        acc = [tip]
        starting_count = tip.count
        made_cut = False
        for i in range(min_length):
            parent = tip.parent
            if parent is None:
                break
            # found branch point:
            if len(parent.children) > 1:
                count_diff = parent.count - starting_count
                if count_diff > max_count_diff:
                    #cut the twig
                    parent.unlink(tip)
                    made_cut =True
                    break
            else:
                tip = parent
                acc.append(tip)
        if not made_cut:
            acc = []
        return acc


class Tree(dict):
    @property
    def nodes(self):
        return self.values()

    @property
    def tips(self):
        return [n for n in self.nodes if not len(n.children)]

    @property
    def roots(self):
        return [n for n in self.nodes if not n.parent]

    @property
    def bifurcations(self):
        return [n for n in self.nodes if len(n.children)>1]
    
    @property
    def total_wiring_length(self):
        visited = set()
        total_length=0
        for tip in self.tips:
            for node in tip.follow_to_root():
                loc = tuple(node.v)
                if loc in visited or not node.parent:
                    continue
                total_length += eu_dist(node.v, node.parent.v)
                visited.add(loc)
        return total_length

    def dfs_traverse(self, **kwargs):
        acc = dict()
        for root in self.roots:
            acc[tuple(root.v)] = root.dfs_subtree(**kwargs)
        return acc
    

    def count_occurences(self, shape=None, progress_bar=False):
        for n in self.nodes:
            n.count = 0
            
        if shape is not None:
            counts =  np.zeros(shape)
        else:
            counts = None
        
        for tip in tqdm(self.tips, disable=not progress_bar):
            for n in tip.follow_to_root():
                if hasattr(n, 'count'):
                    n.count += 1
                else:
                    n.count = 1
                if counts is not None:
                    counts[tuple(n.v)] += 1
        return counts

    def assign_diameters(self, min_diam=0.01, max_diam=6, gamma=1.0,
                        progress_bar=False):
        
        for n in self.nodes:
            n.diam = 0
            
        for tip in tqdm(self.tips, disable=not progress_bar):
            for n in tip.follow_to_root():
                if not hasattr(n, 'diam'):
                    n.diam = 0
                n.diam += min_diam**gamma
        
        for n in self.nodes:
            n.diam = min(max_diam, n.diam**(1/gamma))

    def get_tortuosity(self, tips_only=True):
        nodes = self.tips if tips_only else self.nodes
        return [n.tortuosity for n in nodes]
    
                
    def get_wriggliness(self, tips_only=True):
        simple_tree = self.get_simple()

        simple_tree.add_morphometry()
        
        if tips_only:
            simple_locs = [tuple(n.v) for n in simple_tree.tips]
        else:
            simple_locs = list(simple_tree.keys())
        
        return [self[loc].tortuosity - simple_tree[loc].tortuosity
                for loc in simple_locs]

    def add_morphometry(self, with_bif_angle=False):
        self.count_occurences()
        subtrees = self.dfs_traverse(fn=lambda m:m)
        for root,nodelist in subtrees.items():
            for node in nodelist:
                node.set_root_loc()
                node.set_path_length()
                node.set_root_angle()
                node.set_tortuosity()
                node.root_distance = eu_dist(node.v, node.root_v)
                if with_bif_angle and (node.parent is not None) and (len(node.children)>1):
                    #print(f'setting bifurcation angle for {node}')
                    node.set_bifurcation_angle()



    def copy(self, verbose=False, scale=None, shift=0):
        new_tree = Tree()
        loc = next(iter(self.keys()))
        ndim = len(loc)
        if scale is None:
            scale = np.ones(ndim)
        elif np.iterable(scale):
            scale = scale[:ndim]
        else:
            scale = np.ones(ndim)*scale
            
        for loc, node in tqdm(self.items(), disable=not verbose):
            new_loc = tuple(np.array(loc)*scale + shift)
            newnode = PathNode(new_loc)
            
            if hasattr(node, 'diam'):
                newnode.diam = node.diam
            if hasattr(node, 'count'):
                newnode.count = node.count
                
            new_tree[new_loc] =  newnode   
        
        for loc,node in tqdm(self.items(), disable=not verbose):
            new_loc = tuple(np.array(loc)*scale + shift)
            newnode = new_tree[new_loc]
            if node.parent is not None:
                #ploc = tuple(node.parent.v)
                new_ploc = tuple(node.parent.v*scale + shift)
                new_tree[new_ploc].link(newnode)
                
        return new_tree
            
    def get_simple(self, whitelist=None):
        if whitelist is None:
            whitelist = set()
        new_tree = Tree()
        for tip in self.tips:
            loc = tuple(tip.v)
            active = PathNode(loc)
            new_tree[loc] = active
            for node in tip.follow_to_root():
                if node == tip:
                    continue
                loc = tuple(node.v)
                if (loc in whitelist) \
                    or (len(node.children) != 1) \
                    or (not node.parent):
                    spawn = new_tree.get(loc, PathNode(loc))
                    spawn.link(active)
                    new_tree[loc]=spawn
                    active = spawn
        return new_tree                       
    
    def get_backbone(self, min_count = 100):
        xtree = self.copy()
        xtree.count_occurences()
        good_nodes = [n for n in xtree.nodes if n.count >= min_count]

        unlinked = []
        for node in good_nodes:
            for ch in node.children:
                if ch.count < min_count:
                    node.unlink(ch)
                    unlinked.append(ch)

        return Tree({tuple(n.v):n for n in good_nodes})



    def prune_twigs(self, min_length=10, max_count_diff=10,verbose=False):
        xtree = self.copy()
        if verbose:
            print('copied tree')
        twigs = []
        for tip in xtree.tips:
            twig = tip.prune_twig(min_length=min_length, max_count_diff=max_count_diff)
            if len(twig):
                twigs.append(twig)
        if verbose:
            print(f'cut {len(twigs)} twigs, adjusting tree links')
        twig_tree = Tree()
        for twig in twigs:
            for p in twig:
                twig_tree[tuple(p.v)] = p
        
        clean_tree = Tree()
        for loc, p in xtree.items():
            if len(p.follow_to_root()) > min_length:
                if not loc in twig_tree:
                    for child in p.children:
                        if tuple(child.v) in twig_tree:
                            p.unlink(child)
                    clean_tree[loc] = p
        for tip in clean_tree.tips:
            for p in tip.follow_to_root():
                loc = tuple(p.v)
                if not loc in clean_tree:
                    clean_tree[loc] = p
        if verbose:
            print('old tree size:', len(tree))
            print('new tree size:', len(clean_tree))
        return clean_tree, twig_tree



def get_branching_pattern(tree,nsteps=100,normed=True):
    nodecoll = tree.dfs_traverse(fn=lambda m:m) 
    nodes = reduce(op_.add, nodecoll.values())

    for n in nodes:
        n.set_root_loc()
        n.root_distance = eu_dist(n.v, n.root_v)
    

    bif_dists = np.array([n.root_distance for n in tree.bifurcations])
    tip_dists = np.array([n.root_distance for n in tree.tips])
    max_r = np.max([n.root_distance for n in nodes])
    radii = np.linspace(0,max_r, nsteps)
    bp = np.array([np.sum(bif_dists <=r) - np.sum(tip_dists<=r) 
                     for r in radii])
    if normed:
        bp = bp/(len(tree.bifurcations) + len(tree.tips))
    return radii, bp



# +

def add_fake_z(tree2d):
    tree_3d = Tree()
    for loc, n in tree2d.items():
        loc3d = (0,)+loc
        n.v = np.array(loc3d)
        tree_3d[loc3d] = n
    return tree_3d
    
def to_swc_table(tree, ttype=7, verbose=False, add_fake_soma=True):
    "convert a tree to swc table"
    # start from 2 to allow for soma
    cell_ids = {loc:j+2 for j,loc in enumerate(tree)}
    acc = []
    visited = set()

    roots = tree.roots
    soma_entry=None
    if add_fake_soma:
        center_loc = np.mean([r.v for r in roots],0)
    
        soma_radius = np.max([r.diam for r in roots])*1.5/2
        soma_entry =  dict(idx=1, ttype=1, 
                           x=center_loc[1], y=center_loc[2], z=center_loc[0], 
                           radius=soma_radius, parent = -1)
        visited.add(tuple(center_loc))


    
    for tip in tree.tips:
        for p in tip.follow_to_root():
            loc = tuple(p.v)
            if verbose:
                print(tip, p, loc)
            if loc in visited:
                continue
            my_id = cell_ids[loc]
            parent = p.parent
            
            parent_id = cell_ids[tuple(parent.v)] if parent else -1
            radius = p.diam/2 if hasattr(p, 'diam') else 0.125
            row = dict(idx=my_id,
                       ttype=ttype if parent else 1, 
                       x = loc[1],
                       y = loc[2],
                       z = loc[0],
                       radius = radius,
                       parent = parent_id)
            acc.append(row)
            visited.add(loc)
    acc = acc[::-1] # parents come before children
    if soma_entry is not None:
        acc.append(soma_entry)
    return pd.DataFrame(acc)

def from_swc_table(df, useZ = True):
    tree_tmp = dict()
    for j,row in df.iterrows():
        loc = row.x,row.y,row.z
        if not useZ:
            loc = loc[:-1]
        node = PathNode(loc)
        node.diam = row.radius*2
        tree_tmp[int(row.idx)] = node

    tree = dict()
    for j,row in df.iterrows():
        n = tree_tmp[int(row.idx)]
        tree[tuple(n.v)] = n
        if row.parent >=0:
            pn = tree_tmp[int(row.parent)]
            pn.link(n)
            
    return tree

def load_swc(path, query=None):
    dfx = pd.read_csv(path, comment='#',sep=' ', 
                      skipinitialspace=True,
                      header=None)
    dfx.columns = ['idx','ttype','x','y','z','radius', 'parent']
    if query is not None:
        dfx = dfx.query(query)
    return from_swc_table(dfx)


def save_swc_table(path, df):
    "save pandas DataFrame with swc data to a file in '.swc' format"
    with open(path, 'w') as storage:
        storage.write('#iFM2B3D structure\n')
        storage.write('#id type x y z radius parent\n')
        df.to_csv(storage, sep=' ', header=False, index=False)
        
#def follow_to_root_rec(tip):
#        return [tip] + ([] if not tip.parent else follow_to_root(tip.parent))



# def count_occurences_nx(G, shape):
#     counts =  np.zeros(shape)
#     for p in G:
#         n = G.nodes[p]
#         n['count'] = 0
        
#     for tip in tqdm(get_tips_nx(G)):
#         for p in follow_to_root_nx(G,tip):
#             n = G.nodes[p]
#             if 'count' in n:
#                n['count'] += 1
#             else:
#                n['count'] = 1
#             counts[p] += 1
#     return counts




# def assign_diameters_nx(G, min_diam=0.01, max_diam=6, gamma=1.0):
#     for n in G:
#         G.nodes[n]['diam'] = 0
        
#     for tip in tqdm(get_tips_nx(G)):
#         for p in follow_to_root_nx(G,tip):
#             n = G.nodes[p]
#             n['diam'] += min_diam**gamma
#     for n in G:
#         G.nodes[n]['diam'] = min(max_diam, G.nodes[n]['diam']**(1/gamma))

