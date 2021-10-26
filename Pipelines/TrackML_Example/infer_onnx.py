"""Use onnxruntime to exectute the inference pipeline"""
import os

import numpy as np
import onnxruntime

import torch
from torch import onnx

from torch2onnx import e_onnx_name, f_onnx_name, g_onnx_name, process
from torch2onnx import e_input_name, f_input_name, g_input_name
from torch2onnx import e_output_name, f_output_name, g_output_name

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_id = 0
embedding_dim = 8
r_val = 1.6
knn_val = 500
filter_cut = 0.05


def create_sessions(provider='CPUExecutionProvider'):
    """create onnxruntime sessions, CPUExecutionProvider or CUDAExecutionProvider"""
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads=16
    sess_options.enable_profiling = False

    models = [ onnxruntime.InferenceSession(onnx_name, sess_options, providers=[provider]) 
        for onnx_name in (e_onnx_name, f_onnx_name, g_onnx_name)]
    return models


def build_edges_frnn(spatial, r_val, knn_val):
    import frnn
    dists, idxs, _, _ = frnn.frnn_grid_points(
        points1=spatial.unsqueeze(0), points2=spatial.unsqueeze(0),
        lengths1=None, lengths2=None, K=knn_val, r=r_val, grid=None,
        return_nn=False, return_sorted=True)

    # Remove the unneccessary batch dimension
    idxs = idxs.squeeze()

    ind = torch.Tensor.repeat(torch.arange(idxs.shape[0], device=device), (idxs.shape[1], 1), 1).T
    positive_idxs = idxs >= 0
    edge_list = torch.stack([ind[positive_idxs], idxs[positive_idxs]])
    dist_list = dists.squeeze()[positive_idxs]

    # Remove self-loops and those outsize radius
    # edge_list = edge_list[:, (edge_list[0] != edge_list[1]) & (dist_list <= r_val)]
    return edge_list


def build_edges_faiss(spatial, r_val, knn_val):
    import faiss
    if type(spatial) == torch.Tensor:
        spatial = spatial.cpu().numpy()

    if device == "cuda":
        res = faiss.StandardGpuResources()
        D, I = faiss.knn_gpu(res, spatial, spatial, knn_val)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        D, I = index.search(spatial, knn_val)

    D, I = torch.Tensor(D).to(device), torch.Tensor(I).to(device)
    ind = torch.Tensor.repeat(torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1).T
    edge_list = torch.stack([ind[D <= r_val**2], I[D <= r_val**2]])
    edge_list = edge_list.type(torch.long)
    
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list

build_edges = build_edges_faiss

def run_session_with_iobinding(
    sess, input_data_map, output_name, output_shape,
    element_type=np.float32):

    # OnnxRuntime will copy the data over to the CUDA device 
    # if 'input' is consumed by nodes on the CUDA device
    io_binding = sess.io_binding()

    for key,val in input_data_map.items():
        x_ort_val = onnxruntime.OrtValue.ortvalue_from_numpy(
            val, device_type=device, device_id=device_id)
        io_binding.bind_ortvalue_input(key, x_ort_val)

    y_ort_val = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        output_shape, element_type, device)
    io_binding.bind_ortvalue_output(output_name, y_ort_val)

    sess.run_with_iobinding(io_binding)
    return io_binding.copy_outputs_to_cpu()[0]


# following code is largely taken from 
# https://github.com/exatrkx/exatrkx-work/blob/libtorch-frnn/Inference/notebooks/inferenceOnnxFrnnGnn.py
def inference_onnx(in_data):
    # in_data should be [r, phi, z]. Order matters.
    if device == "cuda":
        models = create_sessions('CUDAExecutionProvider')
    else:
        models = create_sessions()

    # Embedding
    e_sess, f_sess, g_sess = models
    e_input_data = {
        e_input_name[0]: in_data
    }
    batch_size = in_data.shape[0]
    spatial = run_session_with_iobinding(
        e_sess, e_input_data, e_output_name[0], 
        output_shape=(batch_size, embedding_dim))

    # <NOTE> if device is gpu, the movement is useless.
    spatial = torch.tensor(spatial).to(device)

    e_spatial = build_edges(spatial, r_val=r_val, knn_val=knn_val)
    in_data = torch.tensor(in_data, device=device)
    r_dist = torch.sqrt(in_data[:, 0]**2 + in_data[:, 2]**2)
    e_spatial = e_spatial[:, (r_dist[e_spatial[0]] <= r_dist[e_spatial[1]])]
    # print("ONNX:", e_spatial.shape)

    # Filtering
    f_input_data = {
        f_input_name[0]: in_data.cpu().numpy(),
        f_input_name[1]: e_spatial.cpu().numpy(),
    }
    output = run_session_with_iobinding(
        f_sess, f_input_data, f_output_name[0],
        output_shape=(e_spatial.shape[1], 1))
    output = torch.FloatTensor(output).squeeze()
    output = torch.sigmoid(output)
    edge_list = e_spatial[:, output >= filter_cut]
    # print("ONNX:", edge_list.shape)

    # GNN
    g_input_data = {
        g_input_name[0]: in_data.cpu().numpy(),
        g_input_name[1]: edge_list.cpu().numpy()
    }
    gnn_output = run_session_with_iobinding(
        g_sess, g_input_data, g_output_name[0],
        output_shape=(edge_list.shape[1],))
    gnn_output = torch.FloatTensor(gnn_output)
    gnn_output = torch.sigmoid(gnn_output)
    # print("ONNX", gnn_output[gnn_output > 0.4].shape)
    # print("ONNX", gnn_output[gnn_output > 0.4])    

    return spatial.cpu().numpy(), output.cpu().numpy(), gnn_output.cpu().numpy(), edge_list.cpu().numpy()


def inference_model(in_data):
    from torch2onnx import load_models
    models = load_models()
    [m.to(device) for m in models]
    e_model, f_model, g_model = models


    in_data = torch.FloatTensor(in_data).to(device)
    with torch.no_grad():
        spatial = e_model(in_data)
    
    e_spatial = build_edges(spatial, r_val=r_val, knn_val=knn_val)
    r_dist = torch.sqrt(in_data[:, 0]**2 + in_data[:, 2]**2)
    e_spatial = e_spatial[:, (r_dist[e_spatial[0]] <= r_dist[e_spatial[1]])]

    # print(e_spatial.shape)

    with torch.no_grad():
        output = f_model(in_data, e_spatial)
    output = output.squeeze()
    output = torch.sigmoid(output)
    edge_list = e_spatial[:, output >= filter_cut]

    # print(edge_list.shape)

    with torch.no_grad():
        gnn_output = g_model(in_data, edge_list)
    gnn_output = torch.sigmoid(gnn_output)
    # print(gnn_output[gnn_output > 0.4].shape)
    # print(gnn_output[gnn_output > 0.4])

    return spatial.cpu().numpy(), output.cpu().numpy(), gnn_output.cpu().numpy(), edge_list.cpu().numpy()


def tracks_from_gnn(hit_id, score, senders, receivers,
    edge_score_cut=0., epsilon=0.25, min_samples=2, **kwargs):

    import scipy as sp
    from sklearn.cluster import DBSCAN
    import pandas as pd 
    n_nodes = hit_id.shape[0]
    if edge_score_cut > 0:
        cuts = score > edge_score_cut
        score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]
        
    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sp.sparse.csr_matrix((score, (senders, receivers)),
        shape=(n_nodes, n_nodes), dtype=np.float32)
    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1]/2.
    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data
    # make it symmetric
    e_csr_bi = sp.sparse.coo_matrix(
        (np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]), 
        np.hstack([np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),                                                                   
        np.vstack([e_csr.tocoo().col, e_csr.tocoo().row])])))

    # DBSCAN get track candidates
    clustering = DBSCAN(
        eps=epsilon, metric='precomputed',
        min_samples=min_samples).fit_predict(e_csr_bi)
    track_labels = np.vstack(
        [np.unique(e_csr_bi.tocoo().row),
        clustering[np.unique(e_csr_bi.tocoo().row)]])
    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]
    new_hit_id = np.apply_along_axis(
        lambda x: hit_id[x], 0, track_labels.hit_id.values)
    tracks = pd.DataFrame.from_dict(
        {"hit_id": new_hit_id, "track_id": track_labels.track_id})
    return tracks


def process_one_evt(evtid, indir, outdir, **kwargs):

    outdir_onnx = os.path.join(outdir, "tracks_from_onnx")
    os.makedirs(outdir_onnx, exist_ok=True)
    outdir_model = os.path.join(outdir, "tracks_from_models")
    os.makedirs(outdir_model, exist_ok=True)

    out_onnx_name = os.path.join(outdir_onnx, f"{evtid}.npz")
    out_model_name = os.path.join(outdir_model, f"{evtid}.npz")
    if os.path.exists(out_onnx_name) and os.path.exists(out_model_name):
        print(f"{evtid} is there, skip.")
        return


    filename = f'{indir}/{evtid}'
    data = torch.load(filename, map_location=device)
    input_data = data.x.cpu().numpy()

    scales = np.array([3000, np.pi, 400], dtype=np.float32)
    
    res_model = inference_model(input_data / scales)
    res_onnx = inference_onnx(input_data / scales)


    hid = data.hid.cpu().numpy()

    # tracks from native model
    reco_tracks = tracks_from_gnn(
        hid, res_model[2], res_model[3][0], res_model[3][1], **kwargs)
    np.savez(out_model_name, predicts=reco_tracks)

    # tracks from onnx model
    reco_tracks_onnx = tracks_from_gnn(
        hid, res_onnx[2], res_onnx[3][0], res_onnx[3][1], **kwargs)
    np.savez(out_onnx_name, predicts=reco_tracks_onnx)


if __name__ == '__main__':
    import argparse
    import glob
    from functools import partial
    from multiprocessing import Pool
    import multiprocessing as mp
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description='Onnxruntime based inference.')
    add_arg = parser.add_argument
    add_arg('-i', "--indir", help='processed ACTS data', required=True)
    add_arg('-o', "--outdir", help='output directory', required=True)
    add_arg("--max-evts", help='maximum number of events', type=int, default=1)
    add_arg("--num-workers", help='number of threads', default=1, type=int)

    # hyperparameters for DB scan
    add_arg("--epsilon", help='epsilon in DBScan', default=0.4, type=float)
    add_arg("--min-samples", help='minimum number of samples in DBScan', default=2, type=int)
    args = parser.parse_args()

    indir, outdir = args.indir, args.outdir

    all_files = glob.glob(os.path.join(indir, "*"))
    n_tot_files = len(all_files)
    max_evts = args.max_evts if args.max_evts > 0 and \
        args.max_evts <= n_tot_files else n_tot_files
    print("Out of {} events processing {} events with {} workers".format(
        n_tot_files, max_evts, args.num_workers))
    all_evtids = [int(os.path.basename(x)) for x in all_files]

    if args.num_workers > 1:
        # Faiss GPU raised an error 
        # Faiss assertion 'blasStatus == CUBLAS_STATUS_SUCCESS' failed.
        with Pool(args.num_workers) as p:
            process_fnc = partial(process_one_evt, **vars(args))
            p.map(process_fnc, all_evtids[:max_evts])
    else:
        for evtid in all_evtids[:max_evts]:
            process_one_evt(evtid, **vars(args))