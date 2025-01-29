from dataclasses import dataclass

@dataclass
class Args:
    total_timesteps:int = 1000000
    buffer_size: int = 1024
    episod_length:int = 60
    gamma:float = 0.9
    gae_lambda:float = 0.95
    learning_rate:float = 0.0001
    lr_decay:bool = True
    lr_final:float = 1e-08
    max_pool:bool = False
    d_model:int = 32
    in_features:int = 15
    dim_feedforward:int = 512
    nhead: int = 4
    rep_length:int = 12
    clip_range:float = 0.1
    n_epochs:int = 3
    vf_coef: float = 0.5
    ent_decay: bool = False
    ent_coef:float = 0.001
    batch_size:int = 5
    grad_clip:bool = True
    max_grad: float = 10
    clip_vf:float= True
    rwd_scale:float = 1
    seed:int = 5004562
    normalize_advantage: bool = True
    norm_first: bool = True
    policy:str = "Policy_NEmbRole_v4"
    logs_path:str = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/runs/Policy_NEmbRole_v4-FixReport-12_10-Seed_5004562"#"/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/Hostclinet/Run-Policy_EmbRole-clip1-ent1"
    env_path:str = "/home/amine.andam/HostClient/env/FixReport/AppLinux.x86_64" #"/home/amine.andam/HostClient/env/SameReaward/AppLinux.x86_64"
    save_path:str = "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/model/Policy_NEmbRole_v4-FixReport-12_10-Seed_5004562"
    device: str = "cpu" #"cuda" 
