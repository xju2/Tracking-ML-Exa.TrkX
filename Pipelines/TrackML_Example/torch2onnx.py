#!/usr/bin/env python
"""This module converts pytorch models into onnx models"""

import os
import torch


device = 'cpu'

###############################
# prepare input checkpoints
###############################
models_dir = '/home/xju/ocean/lrt/trained_models_v0'
post_fix = 'trained/checkpoints/last.ckpt'

embed_dname = 'Embedding'
embed_ckpt_dir = os.path.join(models_dir, embed_dname, post_fix)

filter_dname = 'Filter'
filtering_ckpt_dir = os.path.join(models_dir, filter_dname, post_fix)

gnn_dname = 'GNN'
gnn_ckpt_dir = os.path.join(models_dir, gnn_dname, post_fix)

###############################
# prepare output onnx models
###############################
outdir = 'onnx_models'
e_onnx_name = os.path.join(outdir, 'embedding.onnx')
f_onnx_name = os.path.join(outdir, 'filtering.onnx')
g_onnx_name = os.path.join(outdir, 'gnn.onnx')

# hardcoded input names and output names
# for emebeding, filtering, and GNN.
e_input_name = ["sp_features"]
e_output_name = ["embedding_output"]
e_dynamic_axes = {
    e_input_name[0]: [0]
}

f_input_name = ["f_nodes", "f_edges"]
f_output_name = ["f_edge_score"]
f_dynamic_axes = {
    f_input_name[0]: [0],
    f_input_name[1]: [1],
}

g_input_name = ["g_nodes", "g_edges"]
g_output_name = ["gnn_edge_score"]
g_dynamic_axes = {
    g_input_name[0]: [0],
    g_input_name[1]: [1],
}

# input and output dimentions for each step
e_input_size = 3

def load_models():
    from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
    e_ckpt = torch.load(embed_ckpt_dir, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_model = LayerlessEmbedding(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.eval()

    from LightningModules.Filter.Models.vanilla_filter import VanillaFilter
    f_ckpt = torch.load(filtering_ckpt_dir, map_location=device)
    f_config = f_ckpt['hyper_parameters']
    f_config['regime'] = []
    # print(f_config['regime'])
    f_model = VanillaFilter(f_config).to(device)
    f_model.load_state_dict(f_ckpt['state_dict'])
    f_model.eval()

    from LightningModules.GNN.Models.agnn import ResAGNN
    g_ckpt = torch.load(gnn_ckpt_dir, map_location=device)
    g_config = g_ckpt['hyper_parameters']
    g_model = ResAGNN(g_config)
    g_model.load_state_dict(g_ckpt['state_dict'])
    g_model.eval()

    return [e_model, f_model, g_model]


def process():
    os.makedirs(outdir, exist_ok=True)

    onnx_config = dict(opset_version=12,
                    export_params=True,
                    do_constant_folding=True)

    batch_size = 1

    # Load all models
    e_model, f_model, g_model = load_models()


    # Embedding
    dummy_input = torch.randn(batch_size, e_input_size, requires_grad=True)

    torch.onnx.export(e_model, dummy_input,
                    e_onnx_name,
                    input_names=e_input_name,
                    output_names=e_output_name,
                    dynamic_axes=e_dynamic_axes, **onnx_config)
    print("Embedding is done")


    # Filtering
    dummy_node_input = torch.ones(batch_size*2, e_input_size, requires_grad=False, device=device)
    dummy_edge_input = torch.ones(2, batch_size, requires_grad=False, device=device, dtype=torch.long)
    print(dummy_edge_input.shape)
    dummy_input = (dummy_node_input, dummy_edge_input)


    torch.onnx.export(f_model, dummy_input,
                    f_onnx_name,
                    input_names=f_input_name,
                    output_names=f_output_name,
                    dynamic_axes=f_dynamic_axes,
                    **onnx_config)
    print("Filtering is done")

    # GNN
    torch.onnx.export(g_model, dummy_input, g_onnx_name,
                    input_names=g_input_name,
                    output_names=g_output_name,
                    dynamic_axes=g_dynamic_axes,
                    **onnx_config)
    print("GNN is done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert trained ExaTrkx model to onnx')
    add_arg = parser.add_argument
    args = parser.parse_args()
    process()