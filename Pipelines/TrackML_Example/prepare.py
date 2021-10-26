import glob
import re
import os

import pandas as pd
import numpy as np
import itertools

from torch_geometric.data import Data
import torch

from multiprocessing import Pool
from functools import partial

def true_edges(hits):
    hit_list = hits.groupby(['particle_id', 'geometry_id'], sort=False)['index'].agg(lambda x: list(x)).groupby(level=0).agg(lambda x: list(x))

    e = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    layerless_true_edges = np.array(e).T
    return layerless_true_edges


sp_name = 'spacepoint'
def process(evtid, input_dir, output_dir):
    prefix = os.path.join(input_dir, "event{:09d}".format(evtid))
    hit_fname = "{}-hits.csv".format(prefix)
    measurements_fname = "{}-measurements.csv".format(prefix)
    measurements2hits_fname = "{}-measurement-simhit-map.csv".format(prefix)
    sp_fname = '{}-{}.csv'.format(prefix, sp_name)
    p_name = '{}-particles.csv'.format(prefix)

    hits = pd.read_csv(hit_fname)
    hits = hits[hits.columns[:-1]]
    hits = hits.reset_index().rename(columns = {'index':'hit_id'})

    measurements = pd.read_csv(measurements_fname)
    meas2hits = pd.read_csv(measurements2hits_fname)
    sp = pd.read_csv(sp_fname)
    particles = pd.read_csv(p_name)

    sp_hits = sp.merge(meas2hits, on='measurement_id', how='left').merge(hits, on='hit_id', how='left')
    sp_hits = sp_hits.merge(particles, on='particle_id', how='left')

    r = np.sqrt(sp_hits.x**2 + sp_hits.y**2)
    phi = np.arctan2(sp_hits.y, sp_hits.x)
    sp_hits = sp_hits.assign(r=r, phi=phi)

    sp_hits = sp_hits.assign(R=np.sqrt((sp_hits.x - sp_hits.vx)**2 + (sp_hits.y - sp_hits.vy)**2 + (sp_hits.z - sp_hits.vz)**2))
    sp_hits = sp_hits.sort_values('R').reset_index(drop=True).reset_index(drop=False)

    edges = true_edges(sp_hits)

    scales = np.array([3000, np.pi, 400])
    data = Data(
        x=torch.from_numpy(sp_hits[['r', 'phi', 'z']].values).float(),
        pid=torch.from_numpy(sp_hits['particle_id'].values),
        layers=torch.from_numpy(sp_hits['geometry_id'].values),
        event_file=prefix,
        hid=torch.from_numpy(sp_hits['measurement_id'].values),
        layerless_true_edges=torch.from_numpy(edges)
    )

    filename = os.path.join(output_dir, str(evtid))
    with open(filename, 'wb') as pickle_file:
        torch.save(data, pickle_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="process events in the folder")
    add_arg = parser.add_argument
    add_arg("in_dir", help="input directory")
    add_arg("out_dir", help="output directory")
    add_arg("--pt-min", help='minimum generated particle pT requirements in [MeV]', default=500)
    add_arg("--eta-max", help='maximum generated particle eta requirement', default=4)
    add_arg("-w", "--workers", help='number of workers', default=1, type=int)
    args = parser.parse_args()

    base_dir = args.in_dir
    all_evts = glob.glob(os.path.join(base_dir, "event*-{}.csv".format(sp_name)))
    pattern = "event([0-9]*)-{}.csv".format(sp_name)
    all_evtids = sorted([int(re.search(pattern, os.path.basename(x)).group(1).strip()) for x in all_evts])
    print("total {} events".format(len(all_evtids)))
    os.makedirs(args.out_dir, exist_ok=True)

    fnc =  partial(process, input_dir=args.in_dir, output_dir=args.out_dir)
    with Pool(args.workers) as p:
        p.map(fnc, all_evtids)
