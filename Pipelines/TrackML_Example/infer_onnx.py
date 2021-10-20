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
r_val = 1.0
knn_val = 100
filter_cut = 0.1


def create_sessions(provider='CPUExecutionProvider'):
    """create onnxruntime sessions, CPUExecutionProvider or CUDAExecutionProvider"""
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads=16
    sess_options.enable_profiling = False

    models = [ onnxruntime.InferenceSession(onnx_name, sess_options, providers=[provider]) 
        for onnx_name in (e_onnx_name, f_onnx_name, g_onnx_name)]
    
    # [x.set_providers([provider]) for x in models]

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
    if device == "cuda":
        res = faiss.StandardGpuResources()
        D, I = faiss.knn_gpu(res, spatial, spatial, knn_val)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        D, I = index.search(spatial, knn_val)

    ind = torch.Tensor.repeat(torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1).T
    edge_list = torch.stack([ind[D <= r_val**2], I[D <= r_val**2]])
    # dist_list = D[D <= r_val**2]
    
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


    # io_binding.bind_input(
    #     name=input_name,
    #     device_type=x_ort_val.device_name(),
    #     device_id=device_id, element_type=element_type, 
    #     shape=x_ort_val.shape(), buffer_ptr=x_ort_val.data_ptr())

    # io_binding.bind_output(
    #     name=output_name,
    #     device_type=y_ort_val.device_name(),
    #     device_id=device_id, element_type=element_type,
    #     shape=y_ort_val.shape(), buffer_ptr=y_ort_val.data_ptr())

    # print(x_ort_val.device_name(), x_ort_val.data_type())
    
    

    # io_binding.bind_output('output')
    sess.run_with_iobinding(io_binding)
    # return io_binding.get_outputs()[0]
    return io_binding.copy_outputs_to_cpu()[0]


# following code is largely taken from 
# https://github.com/exatrkx/exatrkx-work/blob/libtorch-frnn/Inference/notebooks/inferenceOnnxFrnnGnn.py
def inference(in_data):
    # in_data should be [r, phi, z]. Order matters.
    if device == "cuda":
        models = create_sessions('CUDAExecutionProvider')
    else:
        models = create_sessions()

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

    # filtering stage requires splitting the data into batches
    f_batch_size = 800_000
    f_loader = torch.split(e_spatial, f_batch_size, dim=1)

    print("in filtering")
    output_list = []
    for _, sample in enumerate(f_loader):
        f_input_data = {
            f_input_name[0]: in_data.cpu().numpy(),
            f_input_name[1]: sample.cpu().numpy(),
        }
        output = run_session_with_iobinding(
            f_sess, f_input_data, f_output_name[0],
            output_shape=(sample.shape[1], 1))
        output_list.append(output)

    output_list = np.concatenate(output_list, axis=None)
    output_list = torch.FloatTensor(output_list)
    output = torch.sigmoid(output_list)

    edge_list = e_spatial[:, output > filter_cut]
    print(edge_list)


def process_one_evt():
    pass



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Onnxruntime based inference.')
    add_arg = parser.add_argument
    # add_arg('', help='')
    
    args = parser.parse_args()
    filename = '/home/xju/ocean/lrt/data/NoPileUp_5K_withTruth_processed/1234'
    data = torch.load(filename, map_location=device)
    print(data)
    inference(data.x.cpu().numpy())