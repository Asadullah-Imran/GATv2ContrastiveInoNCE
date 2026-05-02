import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import AdaS_Overall, info_nce_loss
from .preprocess import adjacent_matrix_preprocessing

class Train_SpatialGlue:
    def __init__(self, 
        data,
        datatype = 'SPOTS',
        device= torch.device('cpu'),
        random_seed = 2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600, 
        dim_input=3000,
        dim_output=64,
        weight_factors = [1, 1, 5] # Updated for AdaS-GNN: [recon1, recon2, contrastive]
        ):
        
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.loss_history = []
        
        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        
        # We only extract the spatial graphs; the model builds the feature graphs dynamically
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        
        # feature
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        # dimensions
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        if self.datatype == 'SPOTS':
           self.epochs = 600 
           self.weight_factors = [1, 1, 5] # Recon1, Recon2, Contrastive
    
    def train(self):
        # Initialize the leaner 2-Node overall model
        self.model = AdaS_Overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        
        lambda_1, lambda_2, lambda_3 = self.weight_factors[0], self.weight_factors[1], self.weight_factors[2]

        print("Starting AdaS-GNN Training...")
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            
            # Forward Pass: We ONLY pass the spatial graphs now!
            results = self.model(self.features_omics1, self.features_omics2, 
                                 self.adj_spatial_omics1, self.adj_spatial_omics2)
            
            # 1. Reconstruction Losses (MSE)
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['recon1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['recon2'])
            
            # 2. Contrastive Loss (InfoNCE)
            self.loss_contrastive = info_nce_loss(results['y1'], results['y2'])
                
            # 3. Total Loss
            loss = (lambda_1 * self.loss_recon_omics1) + \
                   (lambda_2 * self.loss_recon_omics2) + \
                   (lambda_3 * self.loss_contrastive)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
            if epoch == 0:
               print(f'Epoch {epoch} - Total Loss: {loss.item():.4f}')
               
            self.loss_history.append(loss.item())
        
        print(f"Model training finished! Final Loss: {loss.item():.4f}\n")    
    
        # Extraction
        with torch.no_grad():
          self.model.eval()
          results = self.model(self.features_omics1, self.features_omics2, 
                               self.adj_spatial_omics1, self.adj_spatial_omics2)
 
        emb_omics1 = F.normalize(results['y1'], p=2, eps=1e-12, dim=1)  
        emb_omics2 = F.normalize(results['y2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['z'], p=2, eps=1e-12, dim=1)
        
        output = {'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
                  'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
                  'SpatialGlue': emb_combined.detach().cpu().numpy() # Kept key name as 'SpatialGlue' for downstream compatibility
                  }
        
        return output