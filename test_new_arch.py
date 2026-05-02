import torch
import anndata as ad
import numpy as np
import pandas as pd
from SpatialGlue.preprocess import construct_neighbor_graph
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

# Create a small dummy dataset
n_spots = 100
dim_feature = 50

# Dummy spatial coordinates
spatial_coords = np.random.rand(n_spots, 2) * 100

# Dummy omics 1
X1 = np.random.rand(n_spots, dim_feature)
adata1 = ad.AnnData(X=X1)
adata1.obsm['spatial'] = spatial_coords
adata1.obsm['feat'] = X1

# Dummy omics 2
X2 = np.random.rand(n_spots, dim_feature)
adata2 = ad.AnnData(X=X2)
adata2.obsm['spatial'] = spatial_coords
adata2.obsm['feat'] = X2

print("Constructing neighbor graphs...")
data = construct_neighbor_graph(adata1, adata2, datatype='SPOTS', n_neighbors=3)

print("Initializing Train_SpatialGlue with new GATv2 + Contrastive Learning architecture...")
model_trainer = Train_SpatialGlue(
    data=data,
    datatype='SPOTS',
    epochs=10,  # small epoch for testing
    dim_input=dim_feature,
    dim_output=16,
    cl_mode='both',  # test the 'both' contrastive mode
    lambda_cl=0.1
)

print("Starting training...")
output = model_trainer.train()

print("Training finished successfully!")
print("Final Combined Embedding Shape:", output['SpatialGlue'].shape)
