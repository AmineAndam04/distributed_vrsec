from dataclasses import dataclass

@dataclass
class Args:
    total_timesteps:int = 1000000
    buffer_size: int = 1024
    episod_length:int = 60
    gamma:float = 0.9   
    gae_lambda:float = 0.95
    learning_rate:float = 0.00001
    hlearning_rate:float = 0.0001
    lr_decay:bool = True
    lr_final:float = 1e-08
    max_pool:bool = False
    d_model:int = 32
    in_features:int = 15
    dim_feedforward:int = 512
    nhead: int = 4
    rep_length:int = 5
    clip_range:float = 0.2
    n_epochs:int = 5
    vf_coef: float = 0.5
    ent_decay: bool = False
    ent_coef:float = 0.001
    batch_size:int = 5
    grad_clip:bool = True
    max_grad: float = 10
    clip_vf:float= True
    rwd_scale:float = 1.0
    host_weight:float =1
    seed:int = 8451662
    evaluate_freq:int =10
    normalize_advantage: bool = True
    norm_first: bool = True
    policy:str = "role_emb"
    logs_path:str = "Embedded"
    env_path:str = "/AppLinux.x86_64" 
    save_path:str = "model/Policy"
    device: str = "cpu" #"cuda" 
    train:bool = True
    het:bool = False
    env_path_deploy:str = "/home/amine.andam/HostClient/env/Deploy/AppLinux.x86_64"
    
