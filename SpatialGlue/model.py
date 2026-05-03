import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# ==========================================
# PROPOSED INNOVATION 1: Contrastive Loss
# ==========================================
# def info_nce_loss(emb1, emb2, temperature=0.5):
#     """
#     Replaces the 3rd Attention Layer.
#     Pulls matched cells together, pushes unmatched cells apart in latent space.
#     """
#     # Normalize embeddings
#     emb1 = F.normalize(emb1, dim=1)
#     emb2 = F.normalize(emb2, dim=1)
    
#     # Calculate cosine similarity matrix (N cells x N cells)
#     logits = torch.matmul(emb1, emb2.T) / temperature
    
#     # The 'True' matches are the diagonal (Cell 1 RNA vs Cell 1 Protein)
#     labels = torch.arange(logits.size(0)).to(emb1.device)
    
#     # Cross Entropy forces the diagonal to 1 and everything else to 0
#     loss = F.cross_entropy(logits, labels)
#     return loss


# ==========================================
# PROPOSED INNOVATION 1: Contrastive Loss (Optimized)
# ==========================================
def info_nce_loss(emb1, emb2, temperature=0.5, batch_size=256):
    """
    Optimized Contrastive Loss.
    Uses mini-batching to prevent O(N^2) memory bottlenecks and speed up training.
    """
    num_cells = emb1.size(0)
    
    # Only batch if our dataset is larger than the batch size
    if num_cells > batch_size:
        # Generate random indices for our mini-batch
        idx = torch.randperm(num_cells)[:batch_size].to(emb1.device)
        
        # Subset the embeddings
        emb1_batch = emb1[idx]
        emb2_batch = emb2[idx]
    else:
        emb1_batch = emb1
        emb2_batch = emb2
        
    # Normalize embeddings
    emb1_batch = F.normalize(emb1_batch, dim=1)
    emb2_batch = F.normalize(emb2_batch, dim=1)
    
    # Calculate cosine similarity matrix ONLY on the batch
    logits = torch.matmul(emb1_batch, emb2_batch.T) / temperature
    
    # The 'True' matches are the diagonal
    labels = torch.arange(logits.size(0)).to(emb1_batch.device)
    
    # Cross Entropy calculates the loss
    loss = F.cross_entropy(logits, labels)
    return loss


# ==========================================
# PROPOSED INNOVATION 2: Dynamic Encoder
# ==========================================
class AdaS_Encoder(Module):
    def __init__(self, in_feat, out_feat, threshold=0.6):
        super(AdaS_Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.threshold = threshold
        
        # Floor 1: Intermediate hidden state (h)
        self.weight1 = Parameter(torch.FloatTensor(in_feat, 64))
        # Floor 2: Final Latent space (Y)
        self.weight2 = Parameter(torch.FloatTensor(64, out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        
    def forward(self, feat, adj_spatial):
        # 1. Initial GNN pass using the physical spatial graph
        h = torch.mm(feat, self.weight1)
        h = torch.spmm(adj_spatial, h)
        h = F.relu(h)
        
        # 2. The Staircase: Dynamic Edge Updating
        # ========================================================
        # THE SPEED FIX: We use torch.no_grad() and .detach() 
        # This stops PyTorch from calculating 9 million memory-heavy 
        # gradients just to build the graph map.
        # ========================================================
        with torch.no_grad():
            h_norm = F.normalize(h.detach(), p=2, dim=1)
            sim_matrix = torch.mm(h_norm, h_norm.T) 
            
            # Filter out noisy edges
            dynamic_adj = torch.where(sim_matrix < self.threshold, torch.zeros_like(sim_matrix), sim_matrix)
            
            # Row normalization (The fix that solved your 151,000 loss)
            dynamic_adj = F.normalize(dynamic_adj, p=1, dim=1) 
        
        # 3. Final GNN pass using the new, dynamically learned biological graph
        y = torch.mm(h, self.weight2)
        y = torch.mm(dynamic_adj, y) 
        
        return y

class Decoder(Module):
    def __init__(self, in_feat, out_feat):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x                  

class AdaS_Overall(Module):
    """
    The streamlined 2-Encoder, 2-Decoder Architecture
    """
    def __init__(self, dim_in1, dim_out1, dim_in2, dim_out2):
        super(AdaS_Overall, self).__init__()
        
        # 2 Adaptive Encoders
        self.encoder1 = AdaS_Encoder(dim_in1, dim_out1)
        self.encoder2 = AdaS_Encoder(dim_in2, dim_out2)
        
        # 2 Reconstruction Decoders
        self.decoder1 = Decoder(dim_out1, dim_in1)
        self.decoder2 = Decoder(dim_out2, dim_in2)
        
    def forward(self, feat1, feat2, adj_spatial1, adj_spatial2):
        
        # 1. Get Adaptive Representations (Y)
        y1_adaptive = self.encoder1(feat1, adj_spatial1)
        y2_adaptive = self.encoder2(feat2, adj_spatial2)
        
        # 2. Get Reconstructions (X_hat)
        recon1 = self.decoder1(y1_adaptive, adj_spatial1)
        recon2 = self.decoder2(y2_adaptive, adj_spatial2)
        
        # 3. Pure Mathematical Fusion (No attention weights needed!)
        z_integrated = (y1_adaptive + y2_adaptive) / 2.0
        
        return {
            'y1': y1_adaptive,
            'y2': y2_adaptive,
            'z': z_integrated,
            'recon1': recon1,
            'recon2': recon2
        }