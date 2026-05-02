import os
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
import h5py
# Open the custom HDF5 file
f = h5py.File('MOB/ATAC_RNA_Seq_MouseBrain_RNA_ATAC.h5', 'r')

# Extract metadata (convert byte strings to regular strings)
cell_names = [c.decode('utf-8') for c in f['Cell'][:]]
gene_names = [g.decode('utf-8') for g in f['Gene'][:]]
layer_names = [l.decode('utf-8') for l in f['LayerName'][:]]
spatial_coords = f['Pos'][:]

# 1. Create RNA AnnData object
adata_rna = ad.AnnData(X=f['X_RNA'][:])
adata_rna.obs_names = cell_names
adata_rna.var_names = gene_names
adata_rna.obs['Layertype'] = layer_names
adata_rna.obsm['spatial'] = spatial_coords

# 2. Create ATAC AnnData object
# Note: ATAC var_names are just PC/LSI dimensions (0 to 49) since it's already reduced
adata_atac = ad.AnnData(X=f['X_ATAC'][:])
adata_atac.obs_names = cell_names
adata_atac.obs['Layertype'] = layer_names
adata_atac.obsm['spatial'] = spatial_coords

# Close the file when done
f.close()

print("RNA Object:", adata_rna)
print("ATAC Object:", adata_atac)

# Path for Mouse Spleen
rna_path = 'data/Dataset11_Human_Lymph_Node_A1/adata_RNA.h5ad'
pro_path = 'data/Dataset11_Human_Lymph_Node_A1/adata_ADT.h5ad'

adata_omics1 = adata_rna.copy()
adata_omics2 = adata_atac.copy()

print("Data loaded successfully from the separate data folder!")

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

print("variable named unique")
# Specify data type
data_type = 'SPOTS'

# Fix random seed
from SpatialGlue.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)

from SpatialGlue.preprocess import clr_normalize_each_cell, pca

n_components = adata_omics2.n_vars - 1 
# RNA
adata_omics1.obsm['feat'] = pca(adata_omics1, n_comps=n_components)

# Protein
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_components)

from SpatialGlue.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)

print (data)

# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_val=2000
# define model
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
model = Train_SpatialGlue(data, datatype=data_type, device=device, epochval=epoch_val)

# train model
output = model.train()


import matplotlib.pyplot as plt

# Assuming your model training output is saved to a variable named 'output'
# (e.g., output = model.train())
loss_history = output['loss_history']

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history, color='b', linewidth=2)
plt.title('SpatialGlue Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Reconstruction & Correspondence Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']



# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from SpatialGlue.utils import clustering
tool = 'louvain' # mclust, leiden, and louvain
clustering(adata, key='SpatialGlue', add_key='SpatialGlue', n_clusters=9, method=tool, use_pca=True)

# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)
sc.tl.umap(adata)

sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.show()

# visualization
import matplotlib.pyplot as plt
fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
sc.pp.neighbors(adata, use_rep='SpatialGlue', n_neighbors=10)
sc.tl.umap(adata)

sc.pl.umap(adata, color='SpatialGlue', ax=ax_list[0], title='SpatialGlue', s=20, show=False)
sc.pl.embedding(adata, basis='spatial', color='SpatialGlue', ax=ax_list[1], title='SpatialGlue', s=25, show=False)

plt.tight_layout(w_pad=0.3)
plt.show()

