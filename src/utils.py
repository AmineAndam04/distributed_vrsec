import torch
import numpy as np
import random

report_length = 12
one_hot = {
    (0, 1, 0): 3,
    (0, 0, 0): 8,
    (0, 0, 1): 8,
    (0, 1, 1): 3,
    (1, 0, 0): report_length
}

def process_obs(obs):
        """
        Reformat the raw observations received from Unity.
        It removes the padding and determines the action mask.
        roles: 1 if it's the host, zero otherwise
        """
        processed_obs = dict()
        action_masks = dict()
        roles  = []
        for key,obs_ in obs.items():
            if "Host" in key:
                roles.append(1)
            else:
                roles.append(0)
            meet_cond = np.where(obs_ == 99)
            if len(meet_cond[0]) > 1 :
                obs_i,hot_i = meet_cond[0][0],meet_cond[0][1]
            else:
                obs_i,hot_i = meet_cond[0][0],-1
            
            raw_obs = obs_[:obs_i]
            raw_hot = obs_[obs_i+1:hot_i]
             
            i,j = 0,0
            obs_cat = []
            action_mask = []
            while (i <= (len(raw_hot) -3)):
                code = tuple(raw_hot[i:i+3])
                idx = one_hot[code]
                obs_cat.append(np.append(raw_obs[j:j+idx],code))
                if code  in [(0,0,0),(0,1,1)]:
                        action_mask.append(0)
                else:
                        action_mask.append(1)

                #if code == (1, 0, 0):
                    #print(raw_obs[j:j+idx])
                i+=3
                j+=idx
            assert len(obs_cat) == len(raw_hot) //3, "Something is wrong with removing padding"
            processed_obs[key] = obs_cat
            action_masks[key] = action_mask

        return processed_obs,action_masks,roles

def actions_tosend_(actions,reports,action_mask):
    a = actions.detach().cpu().squeeze().numpy()
    rep = reports.detach().cpu().numpy()
    actions_tosend = dict()
    bin_actions = dict()
    for idx, (key, item) in enumerate(action_mask.items()):
        a_ = a[idx][:len(item)]
        bin_actions[key] = a_
        if "Host" not in key:
            rep_ = rep[idx][:len(item)].flatten()
            a_ = np.concatenate((a_,rep_))
        max_ = 11 if "Host" in key else 234
        
        try: 
            a_ = np.concatenate((a_, np.zeros(max_ - a_.shape[0])))
        except:
            print(a_.shape[0])
            print(max_)

        actions_tosend[key] = a_
    return bin_actions,actions_tosend

def process_logprob_value(log_probs,values,keys):
    log_probs = log_probs.detach().cpu().squeeze().numpy()
    values = values.detach().cpu().squeeze().numpy()
    logprob_tostor = dict()
    values_tostore = dict()
    for log_prob,value,key in zip(log_probs,values,keys):
        logprob_tostor[key] = log_prob
        values_tostore[key] = value
    return logprob_tostor,values_tostore

class LinearDecayLR:
    """
    Linear decay of lr:
    lr(t) = slop * t + bias
    Where:
            slop (<0) = (lr(final) - lr(0)) / total steps  
            bias = lr(0)
    """
    def __init__(self, optimizer, lr_start, lr_final, total_steps):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_final = lr_final
        self.total_steps = total_steps


        self.slop = (self.lr_final - self.lr_start) / self.total_steps 
        self.bias = self.lr_start
        self.current_lr = self.lr_start

        # Set the initial learning rate
        self._set_lr(self.lr_start)

    def step(self,step):
        lr_step = self.slop * step + self.bias
        self.current_lr  = lr_step
        self._set_lr(lr_step)

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    def get_current_lr(self):
        return self.current_lr

class LinearDecayENT:
    """
    Linear decay of lr:
    lr(t) = slop * t + bias
    Where:
            slop (<0) = (lr(final) - lr(0)) / total steps  
            bias = lr(0)
    """
    def __init__(self,lr_start=1, lr_final=1, total_steps=1):
        self.lr_start = lr_start
        self.lr_final = lr_final
        self.total_steps = total_steps


        self.slop = (self.lr_final - self.lr_start) / self.total_steps 
        self.bias = self.lr_start
        self.current_lr = self.lr_start

    def step(self,step):
        lr_step = self.slop * step + self.bias
        self.current_lr  = lr_step

    def get_current_lr(self):
        return self.current_lr


# def norm_d(grads,d):
#     norms = [torch.linalg.vector_norm(g,d ) for g in grads]
#     total_norm_d = torch.linalg.vector_norm(torch.stack([norm for norm in norms]), 2)
#     return total_norm_d
def norm_d(grads, d):
    # Filter out None gradients
    valid_grads = [g for g in grads if g is not None]
    
    # Compute the norm for each valid gradient
    norms = [torch.linalg.vector_norm(g, d) for g in valid_grads]
    
    # Stack the norms and compute the total norm
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), 2)
    
    return total_norm_d

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False