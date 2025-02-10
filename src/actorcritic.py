import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InverseLeakyReLU(nn.Module):
    def __init__(self, positive_slope=0.01):
        super(InverseLeakyReLU, self).__init__()
        self.positive_slope = positive_slope

    def forward(self, x):
        return torch.where(x > 0, self.positive_slope * x, x)



class Policy_EmbRole(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.role_embedding = torch.nn.Embedding(2,d_model)
        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
        self.state_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1, 
            activation="gelu",
            norm_first = norm_first,
            batch_first=True)
        self.report_proj =nn.Sequential(nn.Linear(in_features = d_model, out_features = 16),
                                              nn.GELU(),
                                             nn.Linear(in_features = 16, out_features = rep_length)) 
        
        self.decison_layer = nn.Sequential(nn.Linear(in_features = rep_length, out_features = 32),
                                              nn.GELU(),
                                             nn.Linear(in_features = 32, out_features = 1)) 
        self.value = nn.Sequential(
            nn.Linear(d_model,64),
            nn.GELU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask,evaluate= False):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        if evaluate:
            actions = (proba > 0.5).float()
        else:
            actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        entropy = -proba * torch.log(proba + 1e-10) - (1 - proba) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        entropy = entropy * action_mask
        entropy = entropy.sum(dim = 1)
        return log_prob,value,entropy
    def initialize(self):
        for name, param in self.named_parameters():
            if "encoder" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param,np.sqrt(2))
                if "bias" in name :
                    nn.init.constant_(param, 0)
            elif "state_embedding" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.xavier_uniform_(param)

                #nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)



class Policy_NEmbRole(nn.Module):
    """Slightly more powerfull value network and decision layer"""
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        #self.role_embedding = torch.nn.Embedding(2,d_model)
        #self.role_embedding.weight.requires_grad = False
        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.report_proj =nn.Sequential(nn.Linear(in_features = d_model, out_features = 16),
                                             nn.GELU(),
                                             nn.Linear(in_features = 16, out_features = rep_length)) 
        self.decison_layer = nn.Sequential(nn.Linear(in_features = rep_length, out_features = 32),
                                             nn.GELU(),
                                             nn.Linear(in_features = 32, out_features = 1)) 
        self.value = nn.Sequential(
            nn.Linear(d_model,64),
            nn.GELU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask,evaluate=False):
        #role_embed = self.role_embedding(roles)
        #print(obs.shape)
        state_embed =  self.state_embedding(obs)
        #embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        if evaluate:
            actions = (proba > 0.5).float()
        else:
            actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        #role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        #embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        entropy = -proba * torch.log(proba + 1e-10) - (1 - proba) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        entropy = entropy * action_mask
        entropy = entropy.sum(dim = 1)
        return log_prob,value,entropy
    def initialize(self):
        for name, param in self.named_parameters():
            if "encoder" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param,np.sqrt(2))
                if "bias" in name :
                    nn.init.constant_(param, 0)
            elif "state_embedding" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)



class Policy_SkipREmbRole(nn.Module):
    """
    We also add the embedding in the value stream
    """
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.role_embedding = torch.nn.Embedding(2,d_model)
        #self.role_embedding.weight.requires_grad = False
        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
        self.state_norm = nn.LayerNorm(d_model)
        self.skip_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.report_proj =nn.Sequential(nn.Linear(in_features = d_model, out_features = 16),
                                             nn.GELU(),
                                             nn.Linear(in_features = 16, out_features = rep_length)) 
        self.decison_layer = nn.Linear(rep_length,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,32),
            nn.GELU(),
            nn.Linear(32,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask,evaluate=False):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        enc_out = self.skip_norm(enc_out)
        reports = self.report_proj(enc_out + role_embed.unsqueeze(dim  = 1))
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = (enc_out+ role_embed.unsqueeze(dim  = 1)).permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        if evaluate:
            actions = (proba > 0.5).float()
        else:
            actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        enc_out = self.skip_norm(enc_out)
        reports = self.report_proj(enc_out+role_embed.unsqueeze(dim  = 1))
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = (enc_out+ role_embed.unsqueeze(dim  = 1)).permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        entropy = -proba * torch.log(proba + 1e-10) - (1 - proba) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        entropy = entropy * action_mask
        entropy = entropy.sum(dim = 1)
        return log_prob,value,entropy
    def initialize(self):
        for name, param in self.named_parameters():
            if "encoder" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param,np.sqrt(2))
                if "bias" in name :
                    nn.init.constant_(param, 0)
            elif "state_embedding" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)


class Policy_EmbRole_v2(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.role_embedding = torch.nn.Embedding(2,d_model)
        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
        self.state_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1, 
            activation="gelu",
            norm_first = norm_first,
            batch_first=True)
        self.report_proj =nn.Sequential(nn.Linear(in_features = d_model, out_features = 16),
                                              nn.GELU(),
                                             nn.Linear(in_features = 16, out_features = rep_length)) 
        
        self.decison_layer = nn.Sequential(nn.Linear(in_features = rep_length, out_features = 32),
                                              nn.GELU(),
                                             nn.Linear(in_features = 32, out_features = 1)) 
        self.value = nn.Sequential(
            nn.Linear(d_model,256),
            InverseLeakyReLU(),
            nn.Linear(256,256),
            InverseLeakyReLU(),
            nn.Linear(256,64),
            InverseLeakyReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_norm(self.state_embedding(obs))
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = enc_out.permute(0, 2, 1)
        pooled_enc_out = self.pooling(enc_out)
        pooled_enc_out = pooled_enc_out.squeeze(-1)  
        value = self.value(pooled_enc_out)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        entropy = -proba * torch.log(proba + 1e-10) - (1 - proba) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        entropy = entropy * action_mask
        entropy = entropy.sum(dim = 1)
        return log_prob,value,entropy
    def initialize(self):
        for name, param in self.named_parameters():
            if "encoder" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.xavier_uniform_(param,np.sqrt(2))
                if "bias" in name :
                    nn.init.constant_(param, 0)
            elif "state_embedding" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.xavier_uniform_(param)

                #nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)

