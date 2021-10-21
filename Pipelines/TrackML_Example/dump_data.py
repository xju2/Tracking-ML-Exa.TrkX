#!/usr/bin/env python
"""dump torch data"""

import os
import numpy as np
import torch

device = 'cpu'
evtid = 1234
outdir = '/home/xju/ocean/lrt/test'
filename = os.path.join(outdir, 'embedding', str(evtid))
e_data = torch.load(filename, map_location=device)
print(e_data)
print(e_data.x[0])

scales = np.array([3000, np.pi, 400])
print(e_data.x[0]*scales)


f_data = torch.load(os.path.join(outdir, 'filter', str(evtid)),
    map_location=device)
print(f_data)
print(f_data.layerless_true_edges[:, 0], f_data.signal_true_edges[:, 0])


g_data = torch.load(os.path.join(outdir, 'gnn', str(evtid)),
    map_location=device)
print(g_data)
print(g_data.score[g_data.score > 0.4].shape)

def check_graph_intersection():
    from LightningModules.Embedding.utils import graph_intersection
    def get_truth(pid, e_spatial, e_bidir):
        e_spatial_easy_fake = e_spatial[:, pid[e_spatial[0]] != pid[e_spatial[1]]]
        y_cluster_easy_fake = torch.zeros(e_spatial_easy_fake.shape[1])
        
        e_spatial_ambiguous = e_spatial[:, pid[e_spatial[0]] == pid[e_spatial[1]]]
        e_spatial_ambiguous, y_cluster_ambiguous = graph_intersection(e_spatial_ambiguous, e_bidir)
        
        e_spatial = torch.cat([e_spatial_easy_fake.cpu(), e_spatial_ambiguous.cpu()], dim=-1)
        y_cluster = torch.cat([y_cluster_easy_fake.cpu(), y_cluster_ambiguous.cpu()])
        
        return e_spatial, y_cluster

    pid = torch.tensor(e_data.pid)
    true_edges = torch.tensor(e_data.layerless_true_edges)
    e_bidir = torch.cat([true_edges, true_edges.flip(0)], axis=-1)
    data = np.load("spatial.npz")
    e_spatial = data['e_spatial']
    print(e_spatial.shape)
    e_spatial = torch.tensor(e_spatial)

    e_spatial, _ = get_truth(pid, e_spatial, e_bidir)
    print(e_spatial.shape)

from infer_onnx import tracks_from_gnn
hid = e_data.hid
reco_tracks = tracks_from_gnn(
    hid, g_data.score, g_data.edge_index[0], g_data.edge_index[1])

outdir = "tracks_from_ian"
os.makedirs(outdir, exist_ok=True)
np.savez(os.path.join(outdir, f'{evtid}.npz'), predicts=reco_tracks)