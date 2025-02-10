import torch
import numpy as np
import torch.nn.functional as F

import os
import src.utils as utils
from src.buffer import Buffer
import src.pad_mask as pad_mask

class MARL_PPO():
    
    
    def __init__(self,
                 env,
                 args,
                 policy,
                 logger,
                 ):

        self.device = args.device
        self.env = env
        self.total_timesteps = args.total_timesteps
        self.buffer_size = args.buffer_size
        self.episod_length = args.episod_length
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.learning_rate = args.learning_rate
        self.max_pool = args.max_pool
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.n_epochs = args.n_epochs
        self.clip_range = args.clip_range
        self.ent_coef = args.ent_coef
        self.ent_decay = args.ent_decay
        self.vf_coef = args.vf_coef
        self.batch_size = args.batch_size
        self.lr_decay = args.lr_decay
        self.lr_final = args.lr_final
        self.grad_clip = args.grad_clip
        self.max_grad = args.max_grad
        self.save_path = args.save_path
        self.clip_vf =args.clip_vf
        self.rwd_scale = args.rwd_scale
        self.in_features = args.in_features
        self.rep_length = args.rep_length
        self.dim_feedforward = args.dim_feedforward
        self.normalize_advantage =args.normalize_advantage
        self.norm_first = args.norm_first
        self.logger = logger
        self.host_weight = args.host_weight
        #self.evaluate_freq = args.evaluate_freq


        # Initialize the trainer
        self.init_trainer(policy)

    def init_trainer(self,policy):
        _ = self.env.reset()
        self.buffer = Buffer(buffer_size= self.buffer_size,batch_size = self.batch_size,agents = self.env.agents,normalize_advantage=self.normalize_advantage,
                             gae_lambda = self.gae_lambda,gamma = self.gamma,rwd_scale = self.rwd_scale,host_weight=self.host_weight)
        self.policy = policy(self.in_features,self.d_model,self.nhead,self.dim_feedforward,self.rep_length,self.norm_first,self.max_pool) 
        self.policy = self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)
        if self.lr_decay :
            self.scheduler = utils.LinearDecayLR(optimizer=self.optimizer, lr_start = self.learning_rate, 
                                           lr_final = self.lr_final , total_steps = self.total_timesteps)
        self.current_time_step =0
        if self.ent_decay:
            self.ent_schedular = utils.LinearDecayENT( lr_start = self.ent_coef, lr_final = 0.0001 , total_steps = 100000)
        
    
    def update_actor_critic(self):
        """
        Update the actor and critic networks using the currently gathered data in the  buffer.
        """
        self.collect_trajectories()
        self.policy.train(True)
        policy_losses = []
        value_losses = []
        entropy_losses = []
        losses = []
        clip_fractions= []
        gradients = []
        if self.ent_decay:
            self.ent_coef = self.ent_schedular.get_current_lr() 
        if self.lr_decay :
            self.logger.add_scalar("train/lr",self.scheduler.get_current_lr(),self.current_time_step)
        else:
            self.logger.add_scalar("train/lr",self.learning_rate,self.current_time_step)
        self.logger.add_scalar("train/ent_ceof",self.ent_coef,self.current_time_step)
       
        
        
        for epoch in range(self.n_epochs):
            approx_kl_divs= []
            
            for batch_data in self.buffer.generate_batches():
                padded,obs_mask = pad_mask.pad_and_mask(batch_data["obs"])
                action_mask_ = pad_mask.pad_action_(batch_data["action_mask"])
                actions = pad_mask.pad_action_(batch_data["actions"])
                roles = torch.tensor(batch_data["roles"],dtype=torch.int32).to(self.device)
                padded,obs_mask,actions,action_mask_ = padded.to(self.device),obs_mask.to(self.device),actions.to(self.device),action_mask_.to(self.device)
                log_prob,value,entropy = self.policy.evaluate_actions(roles,padded,obs_mask,actions,action_mask_)
                log_prob,value,entropy = log_prob.squeeze(),value.squeeze(),entropy.squeeze()
                old_log_prob = torch.tensor(batch_data["log_prob"],dtype=torch.float32).to(self.device)
                ratio = torch.exp(log_prob - old_log_prob)
                #print("ratio",ratio)

                # clipped surrogate loss
                advantages = torch.tensor(batch_data["advantages"],dtype = torch.float32).to(self.device)
                
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
                policy_losses.append(policy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                
                if self.clip_vf :
                    # No clipping
                    old_values = torch.tensor(batch_data["values"],dtype=torch.float32).to(self.device)
                    values_pred = old_values + torch.clamp(
                        value -old_values, -self.clip_range, self.clip_range)
                else:
                    values_pred = value
                    
                value_target = torch.tensor(batch_data["returns"],dtype=torch.float32).to(self.device)
                value_loss = F.mse_loss(value_target, values_pred)
                value_losses.append(value_loss.item())
                #print("value_loss",value_loss)

                entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())
                #print("loss",loss)

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_divs.append(approx_kl_div.item())
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in self.policy.parameters() ]
                grad_norm_2 = utils.norm_d(grads,2)
                gradients.append(grad_norm_2)
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()
        if self.lr_decay:
            self.scheduler.step(self.current_time_step)
        if self.ent_decay:
            self.ent_schedular.step(self.current_time_step)
        self.logger.add_scalar("train/loss",np.mean(losses),self.current_time_step)
        self.logger.add_scalar("train/policy_loss",np.mean(policy_losses),self.current_time_step)
        self.logger.add_scalar("train/value_loss",np.mean(value_losses),self.current_time_step)
        self.logger.add_scalar("train/entropy_loss",-np.mean(entropy_losses),self.current_time_step)
        self.logger.add_scalar("train/aprox_kl",np.mean(approx_kl_divs),self.current_time_step)
        self.logger.add_scalar("train/clip_frac",np.mean(clip_fractions),self.current_time_step)
        self.logger.add_scalar("train/grad_2",np.mean(gradients),self.current_time_step)
    def train(self):
        while (self.current_time_step < self.total_timesteps):
            #print("Started: ", self.current_time_step)
            self.update_actor_critic()   
            #print("finished an update")
            self.save_model()               


    
    def  collect_trajectories(self):
        self.policy.train(False)
        step = 0
        current_eps_step = 0
        obs = self.env.reset()
        self.buffer.reset()
        while (step < self.buffer_size):
            self.current_time_step +=1
            obs_,action_mask,roles = utils.process_obs(obs)
            #check_report(obs)
            padded,obs_mask = pad_mask.pad_and_mask([obs_[agent] for agent in obs_.keys()])
            action_mask_ = pad_mask.pad_action_([action_mask[agent] for agent in obs_.keys()])
            with torch.no_grad():
                padded,obs_mask,action_mask_ = padded.to(self.device),obs_mask.to(self.device),action_mask_.to(self.device)
                roles = torch.tensor(roles,dtype=torch.int32).to(self.device)
                actions, reports, log_prob,value = self.policy(roles = roles,obs = padded,obs_mask = obs_mask ,action_mask = action_mask_)
                #print("The send reports are: ", reports)
            bin_actions, actions_tosend = utils.actions_tosend_(actions,reports,action_mask)
            log_prob,value = utils.process_logprob_value(log_prob,value,obs_.keys())
            next_obs,rewards, *_  = self.env.step(actions_tosend)
            done = 0 if current_eps_step < (self.episod_length-1) else 1
            if done == 1 :
                rewards = {key: rewards[key] + value.get(key, 0) for key in rewards}
            self.buffer.add(obs_,action_mask,bin_actions,rewards,log_prob,value,done)
            
            obs = next_obs
            if done == 1:
                stats = self.env._side_channel_dict["StatsSideChannel"].get_and_reset_stats()
                if stats["totalNumberOfReports"][-1][0] != 0 :
                    report_precision = stats["numberOfMaliciousReports"][-1][0] / stats["totalNumberOfReports"][-1][0] 
                else :
                    report_precision = 0
                self.buffer.store_stats(report_precision)
                obs = self.env.reset()
                current_eps_step = 0
            step+=1
            current_eps_step+=1
        self.buffer.compute_advantage_and_returns()
        rwd, precs = self.buffer.get_trajectories_stats()
        #print("finished a collect")
        for agent in self.env.agents:
            label = '/'.join(["traj",agent.replace("?team=0?","")])
            self.logger.add_scalar(label,rwd[0][agent],self.current_time_step)
        self.logger.add_scalar("traj/global_ep_rwd",rwd[1],self.current_time_step)
        self.logger.add_scalar("traj/precision",precs,self.current_time_step)
    def save_model(self):
        
        
        sufx = "model_at" + str(self.current_time_step) + ".pt"
        path = os.path.join(self.save_path, sufx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'scheduler_state_dict': self.scheduler.state_dict() if self.lr_decay else None,
            'current_time_step': self.current_time_step,
            #'current_update_step': self.current_update_step,
            #'buffer': self.buffer.__dict__,  # Save buffer attributes as a dictionary
        }, path)
        print(f"Model saved to {path}")
    def load_model(self,load_path):
        self.load_path = load_path
        if os.path.isfile(self.load_path):
            checkpoint = torch.load(self.load_path, map_location=self.device)
            
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            #if self.lr_decay and checkpoint['scheduler_state_dict'] is not None:
            #    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_time_step = checkpoint['current_time_step']
            
            # Update the buffer's attributes
            #self.buffer.__dict__.update(checkpoint['buffer'])
            
            print(f"Model loaded from {self.load_path}")
        else:
            print(f"No model found at {self.load_path}")

