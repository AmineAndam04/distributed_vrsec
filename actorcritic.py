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


class Policy(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first) :
        super().__init__()
        self.state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            InverseLeakyReLU(),
            nn.Linear(128,64),
            InverseLeakyReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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

class Policy_REMB(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first) :
        super().__init__()
        self.role_embedding = torch.nn.Embedding(2,d_model)
        #self.role_embedding.weight.requires_grad = False
        self.state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.layer_norm = nn.LayerNorm(in_features)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        
        value = self.value(enc_out.mean(dim = 1))
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
class Policy_Nrm(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        self.state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.layer_norm = nn.LayerNorm(in_features)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        embed =  self.state_embedding(self.layer_norm(obs))
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        #print(proba)
        proba = F.sigmoid(proba)
        
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        embed =  self.state_embedding(self.layer_norm(obs))
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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
class Policy_(nn.Module):
    def __init__(self,in_features, d_model, nhead,dim_feedforward ) :
        super().__init__()
        self.role_embedding = torch.nn.Embedding(2,in_features)
        self.state_embedding = nn.Linear(in_features=in_features, out_features=in_features)
        self.pre_encoder = nn.Linear(in_features=in_features,out_features=d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  
            dropout=0.1, 
            activation="gelu",   #gelu
            batch_first=True)
        self.decision_layer_1 = nn.Linear(d_model,in_features)
        self.reports = nn.Sequential(
            nn.Linear(d_model,128),
            nn.ReLU(),
            nn.Linear(128,in_features))
        self.decison_layer = nn.Linear(in_features,1)
        #self.pre_value = nn.Linear(d_model,d_model)
        self.value = nn.Sequential(
            nn.Linear(in_features,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        embed = self.pre_encoder(embed)
        #print(embed)
        reports = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.reports(reports)
        proba = self.decison_layer(reports)
        #print(proba)
        proba = F.sigmoid(proba)
        
        value = self.value(reports.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        embed = self.pre_encoder(embed)
        #print(embed)
        reports = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.reports(reports)
        proba = self.decison_layer(reports)
        #print(proba)
        proba = F.sigmoid(proba)
        
        value = self.value(reports.mean(dim = 1))
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


class Policy_Emb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        self.state_embedding = nn.Sequential(#nn.LayerNorm(in_features),
                                            nn.Linear(in_features, d_model),
                                            nn.GELU())
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            batch_first=True)
        self.report_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    #nn.LayerNorm(d_model),
                                    nn.Linear(d_model, rep_length))
        self.decison_layer = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    #nn.LayerNorm(d_model),
                                    nn.Linear(d_model, 1))

        
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU(),
            #nn.LayerNorm(64),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)


class Policy_PostREmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        self.role_embedding = torch.nn.Embedding(2,d_model,max_norm = 1)
        self.state_embedding = nn.Sequential(nn.LayerNorm(in_features),
                                            nn.Linear(in_features, d_model),
                                            nn.GELU())
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            batch_first=True)
        self.report_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    nn.LayerNorm(d_model),
                                    nn.Linear(d_model, rep_length))
        self.decison_layer = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    nn.LayerNorm(d_model),
                                    nn.Linear(d_model, 1))

        
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed + role_embed.unsqueeze(dim  = 1),src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        #print(proba)
        proba = F.sigmoid(proba)
        
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        embed =  self.state_embedding(obs)
        enc_out = self.encoder(embed + role_embed.unsqueeze(dim  = 1),src_key_padding_mask = obs_mask)
        #reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)

class Policy_PreREmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        self.role_embedding = torch.nn.Embedding(2,in_features,max_norm = 1)
        self.state_embedding = nn.Sequential(nn.LayerNorm(in_features),
                                            nn.Linear(in_features, d_model),
                                            nn.GELU())
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            batch_first=True)
        self.report_proj = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    nn.LayerNorm(d_model),
                                    nn.Linear(d_model, rep_length))
        self.decison_layer = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.GELU(),
                                    nn.LayerNorm(d_model),
                                    nn.Linear(d_model, 1))

        
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.GELU(),
            nn.Linear(128,64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        embed =  self.state_embedding(obs + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        #print(proba)
        proba = F.sigmoid(proba)
        
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles)
        embed =  self.state_embedding(obs + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        #reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "decison_layer" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name :
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name and param.dim() > 1:
                    nn.init.orthogonal_(param,0.01)
                if "bias" in name:
                    nn.init.constant_(param, 0)


class Policy_NoSEmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        assert in_features == d_model , "d_model and in_features should be the same"
        
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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

class Policy_NoSEmbREmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        assert in_features == d_model , "d_model and in_features should be the same"
        self.role_embedding = torch.nn.Embedding(2,in_features)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        obs =  obs + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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



class Policy_NoSEmbDifREmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        assert in_features == d_model , "d_model and in_features should be the same"
        self.role_embedding = torch.nn.Embedding(2,in_features)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)[:, :-1, :]
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)[:, :-1, :]
        value = self.value(enc_out.mean(dim = 1))
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

class Policy_SEmbDifREmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first) :
        super().__init__()
        assert in_features == d_model , "d_model and in_features should be the same"
        self.state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.role_embedding = torch.nn.Embedding(2,in_features)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="gelu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            InverseLeakyReLU(),
            nn.Linear(128,64),
            InverseLeakyReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)[:, :-1, :]
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)[:, :-1, :]
        value = self.value(enc_out.mean(dim = 1))
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
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)



class Policy_Sep(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length) :
        super().__init__()
        self.state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.v_state_embedding = nn.Linear(in_features=in_features, out_features=d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.v_encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            batch_first=True)
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(d_model,1)
        self.pooling = nn.AdaptiveMaxPool1d(1) 
        self.value = nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        embed =  self.state_embedding(obs)
        v_embed =  self.v_state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        v_enc_out = self.v_encoder(v_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        v_enc_out = v_enc_out.permute(0, 2, 1)
        v_enc_out = self.pooling(v_enc_out)
        v_enc_out = v_enc_out.squeeze(-1)  
        value = self.value(v_enc_out)
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        embed =  self.state_embedding(obs)
        v_embed =  self.v_state_embedding(obs)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        v_enc_out = self.v_encoder(v_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(enc_out)
        proba = F.sigmoid(proba)
        v_enc_out = v_enc_out.permute(0, 2, 1)
        v_enc_out = self.pooling(v_enc_out)
        v_enc_out = v_enc_out.squeeze(-1)  
        value = self.value(v_enc_out)
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


class Policy_LayerEmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first) :
        super().__init__()
        self.regular_user_state_emb = nn.Linear(in_features, d_model)
        self.host_user_state_emb = nn.Linear(in_features, d_model)
                                 
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        
        self.report_proj = nn.Linear(d_model,rep_length)
        self.decison_layer = nn.Linear(rep_length,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            InverseLeakyReLU(),
            nn.Linear(128,64),
            InverseLeakyReLU(),
            nn.Linear(64,1)
        )
        self.d_model = d_model
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        host_mask = (roles == 1) 
        regular_user_mask = (roles == 0)
        host_output = self.host_user_state_emb(obs[host_mask])  
        regular_user_output = self.regular_user_state_emb(obs[regular_user_mask]) 
        embed = torch.zeros((obs.size(0),obs.size(1),self.d_model))
        embed[host_mask] = host_output
        embed[regular_user_mask] = regular_user_output
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        host_mask = (roles == 1) 
        regular_user_mask = (roles == 0)
        host_output = self.host_user_state_emb(obs[host_mask])  
        regular_user_output = self.regular_user_state_emb(obs[regular_user_mask]) 
        embed = torch.zeros((obs.size(0),obs.size(1),self.d_model))
        embed[host_mask] = host_output
        embed[regular_user_mask] = regular_user_output
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        value = self.value(enc_out.mean(dim = 1))
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
            elif "emb" in name:
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


class Policy_LayerEmbDec(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first) :
        super().__init__()
        self.regular_user_state_emb = nn.Linear(in_features, d_model)
        self.host_user_state_emb = nn.Linear(in_features, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward=dim_feedforward,  #64
            dropout=0.1, 
            activation="relu",   #gelu
            norm_first = norm_first,
            batch_first=True)
        self.regular_user_decison_layer = nn.Linear(rep_length, 1)
        self.host_user_decison_layer = nn.Linear(rep_length, 1)
        self.report_proj = nn.Linear(d_model,rep_length)
        
        self.value = nn.Sequential(
            nn.Linear(d_model,128),
            InverseLeakyReLU(),
            nn.Linear(128,64),
            InverseLeakyReLU(),
            nn.Linear(64,1)
        )
        self.d_model = d_model
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        
        host_mask = (roles == 1) 
        regular_user_mask = (roles == 0)
        host_output = self.host_user_state_emb(obs[host_mask])  
        regular_user_output = self.regular_user_state_emb(obs[regular_user_mask]) 
        embed = torch.zeros((obs.size(0),obs.size(1),self.d_model))
        embed[host_mask] = host_output
        embed[regular_user_mask] = regular_user_output
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        host_logits = self.host_user_decison_layer(reports[host_mask])  
        regular_logits = self.regular_user_decison_layer(reports[regular_user_mask]) 
        logits = torch.zeros((reports.size(0),reports.size(1),1))
        logits[host_mask] = host_logits
        logits[regular_user_mask] = regular_logits
        proba = F.sigmoid(logits)
        value = self.value(enc_out.mean(dim = 1))
        actions = torch.bernoulli(proba)
        log_probs = actions * torch.log(proba + 1e-10) + \
                            (1 - actions) * torch.log(1 - proba + 1e-10)
        log_probs = log_probs * action_mask
        log_prob = log_probs.sum(dim = 1)
        actions = actions * action_mask
        return actions,reports, log_prob,value
    def evaluate_actions(self,roles,obs,obs_mask,actions,action_mask):
        host_mask = (roles == 1) 
        regular_user_mask = (roles == 0)
        host_output = self.host_user_state_emb(obs[host_mask])  
        regular_user_output = self.regular_user_state_emb(obs[regular_user_mask]) 
        embed = torch.zeros((obs.size(0),obs.size(1),self.d_model))
        embed[host_mask] = host_output
        embed[regular_user_mask] = regular_user_output
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)

        host_logits = self.host_user_decison_layer(reports[host_mask])  
        regular_logits = self.regular_user_decison_layer(reports[regular_user_mask]) 
        logits = torch.zeros((reports.size(0),reports.size(1),1))
        logits[host_mask] = host_logits
        logits[regular_user_mask] = regular_logits

        proba = F.sigmoid(logits)
        value = self.value(enc_out.mean(dim = 1))
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
            elif "emb" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "report" in name:
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


class Policy_EmbRole(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        print("roles shape", roles.shape)
        print("roles",roles)
        print("role_embed shape",role_embed.shape)
        print("role_embed",role_embed)
        
        state_embed =  self.state_embedding(obs)
        print("state_embed shape", state_embed.shape)
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
        state_embed =  self.state_embedding(obs)
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
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)

class Policy_DbEmbRole(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out + role_embed.unsqueeze(dim  = 1))
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
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out + role_embed.unsqueeze(dim  = 1))
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

class Policy_NEmbRole(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1)
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
        self.decison_layer = nn.Linear(rep_length,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,32),
            nn.GELU(),
            nn.Linear(32,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        state_embed =  self.state_embedding(obs)
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
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
        state_embed =  self.state_embedding(obs)
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

class Policy_SkipNEmbRole(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out + embed)
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
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out+embed)
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


class Policy_EmbRole_v2(nn.Module):
    """We add the role embed after the encoder"""
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out+role_embed.unsqueeze(dim  = 1))
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = (enc_out+role_embed.unsqueeze(dim  = 1)).permute(0, 2, 1)
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
        state_embed =  self.state_embedding(obs)
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out+role_embed.unsqueeze(dim  = 1))
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = (enc_out+role_embed.unsqueeze(dim  = 1)).permute(0, 2, 1)
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


class Policy_EmbRole_v3(nn.Module):
    """We embed before the state embed"""
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.role_embedding = torch.nn.Embedding(2,in_features)
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
        self.decison_layer = nn.Linear(rep_length,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,32),
            nn.GELU(),
            nn.Linear(32,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
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
        state_embed =  self.state_embedding(obs + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
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


class Policy_EmbRole_v4(nn.Module):
    """Slightly more powerfull value network and decision layer"""
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
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
        state_embed =  self.state_embedding(obs)
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
                nn.init.orthogonal_(param)
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)




class Policy_EmbRole_v5(nn.Module):
    """Like v3 but with layer norm"""
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1) 
        self.role_embedding = torch.nn.Embedding(2,in_features)
        #self.role_embedding.weight.requires_grad = False
        self.layer_norm = nn.LayerNorm(in_features)
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
        self.decison_layer = nn.Linear(rep_length,1)
        self.value = nn.Sequential(
            nn.Linear(d_model,32),
            nn.GELU(),
            nn.Linear(32,1)
        )
        self.initialize()
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.layer_norm(self.role_embedding(roles))
        state_embed =  self.state_embedding(self.layer_norm(obs) + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
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
        role_embed = self.layer_norm(self.role_embedding(roles))
        state_embed =  self.state_embedding(self.layer_norm(obs) + role_embed.unsqueeze(dim  = 1))
        enc_out = self.encoder(state_embed,src_key_padding_mask = obs_mask)
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


class Policy_ContRLEmb(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1)

        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
       
        self.role_embedding = torch.nn.Embedding(2,d_model)
        self.role_embedding.weight.requires_grad = False
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)

        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)[:, :-1, :]
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
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)[:, :-1, :]
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
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)



class Policy_NEmbRole_v4(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
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


class Policy_ContRLEmbFixed(nn.Module):
    def __init__(self, in_features, d_model, nhead,dim_feedforward,rep_length,norm_first,max_pool=False) :
        super().__init__()
        if max_pool == True:
            self.pooling = nn.AdaptiveMaxPool1d(1) 
        else :
            self.pooling = nn.AdaptiveAvgPool1d(1)

        self.state_embedding = nn.Sequential(nn.Linear(in_features = in_features, out_features = d_model),
                                             nn.GELU(),
                                             nn.Linear(in_features = d_model, out_features = d_model)) 
       
        self.role_embedding = torch.nn.Embedding(2,d_model)
        self.role_embedding.weight.requires_grad = False
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)

        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)[:, :-1, :]
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
        role_embed = self.role_embedding(roles).unsqueeze(dim = 1)
        obs = self.state_embedding(obs)
        obs = torch.cat((obs,role_embed),dim = 1)
        mask_emb = torch.tensor([[False]] * obs_mask.size(0))
        obs_mask = torch.cat((obs_mask, mask_emb), dim=1)
        enc_out = self.encoder(obs,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out)
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)[:, :-1, :]
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
            elif "report" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,np.sqrt(2))
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "value" in name:
                if "weight" in name:
                    nn.init.orthogonal_(param,1)
                if "bias" in name:
                    nn.init.constant_(param, 0)
            elif "role_embedding" in name:
                nn.init.orthogonal_(param)


class Policy_SkipREmbRole(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out + role_embed.unsqueeze(dim  = 1))
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
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out+role_embed.unsqueeze(dim  = 1))
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



class Policy_SkipREmbRole_v2(nn.Module):
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
        
    def forward(self,roles,obs,obs_mask,action_mask):
        role_embed = self.role_embedding(roles)
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
        reports = self.report_proj(enc_out + role_embed.unsqueeze(dim  = 1))
        proba = self.decison_layer(reports)
        proba = F.sigmoid(proba)
        enc_out = (enc_out+ role_embed.unsqueeze(dim  = 1)).permute(0, 2, 1)
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
        state_embed =  self.state_embedding(obs)
        embed = state_embed + role_embed.unsqueeze(dim  = 1)
        enc_out = self.encoder(embed,src_key_padding_mask = obs_mask)
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


