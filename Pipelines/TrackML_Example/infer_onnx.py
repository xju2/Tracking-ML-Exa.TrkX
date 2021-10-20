"""Use onnxruntime to exectute the inference pipeline"""

import numpy as np
import onnxruntime

import torch
from torch import onnx

from torch2onnx import e_onnx_name, f_onnx_name, g_onnx_name
from torch2onnx import e_input_name, f_input_name, g_input_name
from torch2onnx import e_output_name, f_output_name, g_output_name

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_id = 0
embedding_dim = 8
r_val = 1.6
knn_val = 20
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
    edge_list = edge_list[:, (edge_list[0] != edge_list[1]) & (dist_list <= r_val)]
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
    
    return edge_list

build_edges = build_edges_frnn

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
    edge_list = e_spatial[:, output > filter_cut]

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
    print(gnn_output[gnn_output > 0.4])


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

    # <TODO: why they are here? Taken from Embedding/model/Models/inference.py>
    # random_flip = torch.randint(2, (e_spatial.shape[1],)).bool()
    # e_spatial[0, random_flip], e_spatial[1, random_flip] = (
    #     e_spatial[1, random_flip],
    #     e_spatial[0, random_flip],
    # )


    with torch.no_grad():
        output = f_model(in_data, e_spatial)
    output = output.squeeze()
    output = torch.sigmoid(output)
    edge_list = e_spatial[:, output > filter_cut]

    with torch.no_grad():
        gnn_output = g_model(in_data, edge_list)
    gnn_output = torch.sigmoid(gnn_output)
    print(gnn_output[gnn_output > 0.4])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Onnxruntime based inference.')
    add_arg = parser.add_argument
    # add_arg('', help='')
    
    args = parser.parse_args()
    filename = '/home/xju/ocean/lrt/data/NoPileUp_5K_withTruth_processed/1234'
    data = torch.load(filename, map_location=device)
    print(data)
    input_data = data.x.cpu().numpy()
    print(input_data[0])
    scales = np.array([3000, np.pi, 400])
    inference_model(data.x.cpu().numpy())