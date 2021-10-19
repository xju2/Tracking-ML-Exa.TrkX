#!/usr/bin/env python
"""This module converts pytorch models into onnx models"""

import os
import torch


device = 'cpu'
models_dir = '/home/xju/ocean/lrt/trained_models_v0'
post_fix = 'trained/checkpoints/last.ckpt'
outdir = 'onnx_models'
e_onnx_name = os.path.join(outdir, 'embedding.onnx')
f_onnx_name = os.path.join(outdir, 'filtering.onnx')
g_onnx_name = os.path.join(outdir, 'gnn.onnx')


def process():
    os.makedirs(outdir, exists_ok=True)

    onnx_config = dict(opset_version=12,
                    export_params=True,
                    do_constant_folding=True)
    batch_size = 1


    # Embedding
    embed_dname = 'Embedding'
    embed_ckpt_dir = os.path.join(models_dir, embed_dname, post_fix)
    e_ckpt = torch.load(embed_ckpt_dir, map_location=device)

    from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding

    e_config = e_ckpt['hyper_parameters']
    e_model = LayerlessEmbedding(e_config)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.eval()


    e_input_size = 3
    dummy_input = torch.randn(batch_size, e_input_size, requires_grad=True)
    dynamic_axes = {"hid_features": [0, 1]}

    torch.onnx.export(e_model, dummy_input,
                    e_onnx_name,
                    input_names=["hid_features"],
                    output_names=["embedding_output"],
                    dynamic_axes=dynamic_axes, **onnx_config)
    print("Embedding is done")


    # Filtering
    filter_dname = 'Filter'
    filtering_ckpt_dir = os.path.join(models_dir, filter_dname, post_fix)
    f_ckpt = torch.load(filtering_ckpt_dir, map_location=device)


    from LightningModules.Filter.Models.vanilla_filter import VanillaFilter
    f_config = f_ckpt['hyper_parameters']
    f_model = VanillaFilter(f_config).to(device)
    f_model.load_state_dict(f_ckpt['state_dict'])
    f_model.eval()

    dummy_node_input = torch.ones(batch_size*2, e_input_size, requires_grad=False, device=device)
    dummy_edge_input = torch.ones(2, batch_size, requires_grad=False, device=device, dtype=torch.long)
    print(dummy_edge_input.shape)
    dummy_input = (dummy_node_input, dummy_edge_input)


    torch.onnx.export(f_model, dummy_input,
                    f_onnx_name,
                    input_names=["doublets"],
                    output_names=["filtering_output"], **onnx_config)
    print("Filtering is done")

    # GNN
    gnn_dname = 'GNN'
    gnn_ckpt_dir = os.path.join(models_dir, gnn_dname, post_fix)
    g_ckpt = torch.load(gnn_ckpt_dir, map_location=device)

    from LightningModules.GNN.Models.agnn import ResAGNN

    g_config = g_ckpt['hyper_parameters']
    g_model = ResAGNN(g_config)
    g_model.load_state_dict(g_ckpt['state_dict'])
    g_model.eval()

    dynamic_axes = {"nodes": [0, 1], "edge_index": [0, 1]}
    torch.onnx.export(g_model, dummy_input, g_onnx_name,
                    input_names=['nodes', 'edge_index'],
                    output_names=['gnn_output'],
                    dynamic_axes=dynamic_axes,
                    **onnx_config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert trained ExaTrkx model to onnx')
    add_arg = parser.add_argument
        
    args = parser.parse_args()
    process()