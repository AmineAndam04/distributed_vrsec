import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
from src.utils import set_seed
from src.env import load_env
from src.ppo_learner import MARL_PPO
from src.sep_ppo_learner import MARL_HEPPO
from src.actorcritic import *
from src.config import Args
import datetime
import os
import glob
from src.evaluate import evaluate_test_env,evaluate_from_checkpoint 
policies = {"role_emb":Policy_EmbRole,"role_em_v2":Policy_EmbRole_v2,"no_role_emb":Policy_NEmbRole,"skip_role_emb":Policy_SkipREmbRole}
def parse_args():
    parser = argparse.ArgumentParser(description="Override configuration values from the command line.")

    # Automatically add arguments based on the dataclass fields
    for field, value in asdict(Args()).items():
        arg_type = type(value)
        if arg_type == bool:
            # Special handling for booleans: allow --flag and --no-flag
            parser.add_argument(f"--{field}", dest=field, action="store_true", help=f"Enable {field}")
            parser.add_argument(f"--no-{field}", dest=field, action="store_false", help=f"Disable {field}")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=value, help=f"Set {field} (default: {value})")

    parser.set_defaults(**asdict(Args()))
    return parser.parse_args()

def main(args):
    if args.seed < 0:
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item() 
    set_seed(args.seed)
    if args.train:
        time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.logs_path =  os.path.join("/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/runs/Last", args.logs_path + "_" + time_token)
        args.save_path = args.save_path + "_" + time_token
        writer = SummaryWriter(args.logs_path) 
        hyperparams = vars(args)
        writer.add_text('Hyperparameters', str(hyperparams), 0)
        policy = policies[args.policy]
        env = load_env(args.env_path,args.seed,rep_length= args.rep_length)
        test_env = load_env(args.env_path,args.seed,rep_length= args.rep_length,worker_id=2)
        if args.het : 
            model = MARL_HEPPO(env,test_env,args,policy,logger=writer)
        else :
            model = MARL_PPO(env,test_env,args,policy,logger=writer)

        list_of_files = glob.glob(os.path.join(args.save_path, '*.pt')) 
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Loading saved model from {latest_file}")
            model.load_model(latest_file)
        else:
            print("Creating new model")
        model.train()
        writer.close()
        env.close()
    else:
        #env = load_env(args.env_path,args.seed,rep_length= args.rep_length)
        evaluate_from_checkpoint(args.env_path)
        #env.close()

    
    

if __name__ == "__main__":
    args = parse_args()    
    main(args=args)