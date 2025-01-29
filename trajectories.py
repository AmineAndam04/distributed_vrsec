from src import utils,pad_mask
def collect_trajectories(
        env,
        buffer,
        policy_net,
        episod_length,
        gamma,
        gae_lambda
):
    buffer_size = buffer.buffer_size
    step = 0
    current_eps_step = 0
    obs = env.reset()
    while (step < buffer_size):
        obs_,action_mask = utils.process_obs(obs)
        padded,obs_mask = pad_mask.pad_and_mask([obs_[agent] for agent in obs_.keys()])
        action_mask_ = pad_mask.pad_action_([action_mask[agent] for agent in obs_.keys()])
        actions, reports, log_prob,value = policy_net(obs = padded,obs_mask = obs_mask ,action_mask = action_mask_)
        bin_actions, actions_tosend = utils.actions_tosend_(actions,reports,action_mask)
        log_prob,value = utils.process_logprob_value(log_prob,value,obs_.keys())
        next_obs,rewards, *_  = env.step(actions_tosend)
        done = 0 if current_eps_step < (episod_length-1) else 1
        buffer.add(obs_,action_mask,bin_actions,rewards,log_prob,value,done)
        obs = next_obs
        if done == 1:
            stats = env._side_channel_dict["StatsSideChannel"].get_and_reset_stats()
            report_precision = stats["numberOfMaliciousReports"][-1][0] / stats["totalNumberOfReports"][-1][0] 
            buffer.store_stats(report_precision)
            obs = env.reset()
            current_eps_step = 0
        step+=1
        current_eps_step+=1
    buffer.compute_advantage_and_returns(gae_lambda= gae_lambda, gamma=gamma)

