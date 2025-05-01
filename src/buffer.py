import numpy as np


class Buffer():
    def __init__(
            self,
            buffer_size,
            batch_size,
            agents,
            gae_lambda,
            gamma,
            normalize_advantage,
            rwd_scale,
            host_weight,
            het = False):
        self.buffer_size = buffer_size
        self.agents = agents
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        self.rwd_scale = rwd_scale
        self.host_weight = host_weight
        self.het  = het
        self.reset()
    def reset(self):
        self.obs = { key: [0] * self.buffer_size for key in self.agents }
        self.action_mask = { key: [0] * self.buffer_size for key in self.agents }
        self.actions = { key: [0] * self.buffer_size for key in self.agents }
        self.log_prob = { key: [0] * self.buffer_size for key in self.agents }
        self.values = { key: [0] * self.buffer_size for key in self.agents }
        self.rewards = { key: [0] * self.buffer_size for key in self.agents }
        self.done = np.zeros((self.buffer_size,),dtype=np.int32)
        self.advantages = { key: [0] * self.buffer_size for key in self.agents }
        self.returns = { key: [0] * self.buffer_size for key in self.agents }
        self.common_reward = [0] * self.buffer_size
        self.stats = []
        self.pos = 0 
        self.num_eps  = 0
    def add(self,obs,action_mask,bin_actions,rewards,log_prob,value,done):
        com_rwd = 0
        self.done[self.pos] = done
        for agent in self.agents:
            self.obs[agent][self.pos] = obs[agent]
            self.action_mask[agent][self.pos] = action_mask[agent]
            self.actions[agent][self.pos] = bin_actions[agent]
            self.log_prob[agent][self.pos] = log_prob[agent]
            self.values[agent][self.pos] = value[agent]
            self.rewards[agent][self.pos] = rewards[agent]
            com_rwd += self.host_weight * rewards[agent] if  "Host" in agent else rewards[agent]
        self.common_reward[self.pos] = com_rwd / (len(self.agents)-1 + self.host_weight)
        self.pos += 1
    def get(self):
        return self.obs, self.action_mask, self.actions,self.log_prob,self.values,self.rewards,self.done,self.advantages, self.returns,self.roles
    def store_stats(self,stats):
        self.stats.append(stats)
        self.num_eps += 1
    def compute_ep_reward_mean(self):
        last_index = 0
        for i in range(len(self.done) - 1, -1, -1):
            if self.done[i] == 1:
                last_index = i
                break
        ep_reward_mean = {key : np.sum(self.rewards[key][: last_index +1]) / self.num_eps for key in self.rewards.keys()}
        ep_reward_global_mean = np.mean(list(ep_reward_mean.values()))
        return ep_reward_mean,ep_reward_global_mean
    def get_trajectories_stats(self):
        return self.compute_ep_reward_mean(), np.mean(self.stats)

    def indiv_gae_gt(self,rewards,values,dones,gae_lambda,gamma):
        advantage = [0] * self.buffer_size
        returns = [0] * self.buffer_size
        adv = 0
        for idx in reversed(range(self.buffer_size)):
            terminal = 1- dones[idx]
            if idx == (self.buffer_size -1):
                next_value =  values[idx]
            else :
                next_value =  values[idx+1]
            delta = self.rwd_scale * rewards[idx] + gamma * next_value * terminal - values[idx]
            adv = delta + gamma * gae_lambda * adv * terminal
            advantage[idx] = adv
            returns[idx] = adv + values[idx]
        if self.normalize_advantage and len(advantage) > 1:
                    advantage = np.array(advantage)
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                    returns = np.array(returns)
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return list(advantage), list(returns)
    

    def compute_advantage_and_returns(self):
        
        for agent in self.agents:
            if self.het :
                normalized = np.array(self.rewards[agent])
                normalized = (normalized - np.mean(normalized))/ (np.std(normalized) + 1e-6)
                adv,ret = self.indiv_gae_gt(normalized,self.values[agent],self.done,self.gae_lambda,self.gamma)
            else: 
                normalized = np.array(self.common_reward)
                normalized = (normalized - np.mean(normalized))/ (np.std(normalized) + 1e-6)
                adv,ret = self.indiv_gae_gt(normalized,self.values[agent],self.done,self.gae_lambda,self.gamma)
            self.advantages[agent] = adv
            self.returns[agent] = ret
    
    def generate_batches(self):
        num_batches = self.buffer_size // self.batch_size
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)  

        for i in range(num_batches):
            batch_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
            batch_data = {
                agent: {
                    'obs': [self.obs[agent][idx] for idx in batch_indices],
                    'action_mask': [self.action_mask[agent][idx] for idx in batch_indices],
                    'actions': [self.actions[agent][idx] for idx in batch_indices],
                    'log_prob': [self.log_prob[agent][idx] for idx in batch_indices],
                    'values': [self.values[agent][idx] for idx in batch_indices],
                    'advantages': [self.advantages[agent][idx] for idx in batch_indices],
                    'returns': [self.returns[agent][idx] for idx in batch_indices]
                } for agent in self.agents
            }
            roles =[[1]* self.batch_size if "Host" in agent else [0]*self.batch_size for agent in self.agents]
            roles =[item for sublist in roles for item in sublist]
            attributes = ['obs','action_mask','actions','log_prob','values','advantages','returns']
            batch_data_flatten = { key: self.flatten_batch(batch_data,key) for key in attributes}
            batch_data_flatten["roles"] = roles
            yield batch_data_flatten
    def flatten_batch(self,batch_data, key):
        a = [ batch_data[agent][key] for agent in self.agents]
        b = [item for sublist in a for item in sublist]
        return b