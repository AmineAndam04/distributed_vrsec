import torch 
import numpy as np
from src.actorcritic import *
import src.utils as utils
from src.buffer import Buffer
import src.pad_mask as pad_mask
from src.env import load_env
#import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def evaluate_test_env(env,policy,rep_length,episod_length,hpolicy=None):
    policy.eval()
    results = []
    ep_rewards = []
    for _ in range(10):
        rewards,stats = collect_stats(env,policy,rep_length=rep_length,episod_length = episod_length,hpolicy = hpolicy)
        metrics = compute_stats(stats)
        results.append(metrics)
        ep_rewards.append(rewards)
    avg_rewards = {key: np.sum([ep[key] for ep in ep_rewards]) for key in env.agents}
    avg_metrics = metric_averages(results)
    return avg_rewards,avg_metrics


def metric_averages(data):
        host_metrics = {}
        avatar_metrics = {}

        for ep in range(len(data)):
            for key, metrics in data[ep].items():
                if "Avatar" in key:
                    for metric, value in metrics.items():
                        try:
                            val = float(value)
                        except Exception:
                            val = 0
                        avatar_metrics.setdefault(metric, []).append(val)
                elif "Host" in key:
                    host_data = data[ep].get("Host", {})
                    for metric, value in host_data.items():
                        try:
                            val = float(value)
                        except Exception:
                            val = 0
                        host_metrics.setdefault(metric, []).append(val)
                else:
                    print("something is wrong")

        # Compute the average metrics using np.nanmean (ignoring NaN values).
        host_avg = {
            metric: np.nan if np.all(np.isnan(filtered_values := np.array(values)[np.array(values) > 0])) 
            else np.nanmean(filtered_values)
            for metric, values in host_metrics.items()
        }

        avatar_avg = {
            metric: np.nan if np.all(np.isnan(filtered_values := np.array(values)[np.array(values) > 0])) 
            else np.nanmean(filtered_values)
            for metric, values in avatar_metrics.items()
        }

        average_metrics = {
            "Host": host_avg,
            "Avatars": avatar_avg
        }
        return average_metrics 
def evaluate_from_checkpoint(env_path,rep_length=5):
    remb_checkpoint = ["Policy_2025-02-21_15-37-51",
                        "Policy_2025-02-19_11-08-58",
                        "Policy_2025-02-15_23-23-58",
                        "Policy_2025-02-21_15-36-25",
                        "Policy_2025-02-21_15-38-44"]
    noremb_checkpoint = ["Policy_2025-02-21_15-37-31",
                        "Policy_2025-02-19_11-09-23",
                        "Policy_2025-02-15_23-23-12",
                        "Policy_2025-02-21_15-36-43",
                        "Policy_2025-02-21_15-39-02"] 
    opt_checkpoints = ["Policy_2025-02-24_14-42-31","Policy_2025-02-24_14-42-53","Policy_2025-02-24_14-42-04","Policy_2025-02-24_14-43-19",
                       "Policy_2025-02-24_14-43-43"]
    max_fusion = [
         "Policy_2025-03-03_11-13-06",
         "Policy_2025-03-03_11-14-32",
         "Policy_2025-03-03_11-13-43",
         "Policy_2025-03-03_11-12-11",
         "Policy_2025-03-03_11-15-12",
    ]
    seed = [152643571,822948974,8451662,734212120, 940905178]
    
    for i in range(len(remb_checkpoint)):
        env = load_env(env_path,seed[i],rep_length= rep_length)
             
        
        max_fusion_policy_net = Policy_EmbRole(
            in_features=15, d_model=32, nhead=4, dim_feedforward=512, 
            rep_length=rep_length, norm_first=True, max_pool=True
        )
        print("Seed: ", seed[i])
        path = "" + max_fusion[i] + "/model_at1000448.pt" 
        print("Loaded : ", path)
        role_checkpoint = torch.load(path)
        max_fusion_policy_net.load_state_dict(role_checkpoint["policy_state_dict"])
        max_fusion_policy_net.eval() 
        policies = [max_fusion_policy_net ]
        names = ["MaxFusion"]
        
        for i,policy in enumerate(policies):
            results = []
            ep_rewards = []
            for _ in range(100):
                rewards,stats = collect_stats(env,policy,rep_length=rep_length,episod_length = 60)
                ep_rewards.append(rewards)
                metrics = compute_stats(stats)
                ep_rewards.append(rewards)
                results.append(metrics)
            print("Policy: " + names[i])
            avg_rewards = {key: np.sum([ep[key] for ep in ep_rewards]) for key in env.agents}
            avg_metrics = metric_averages(results)
            print("average of rewards: ", avg_rewards)
            print("average metrics: ", avg_metrics)
            print("****" * 10)
            break
        print("+++++++++" * 10)
        env.close()
    
    return 0


def print_policy_averages(data):
    """
    Loads the file from the given path and prints the average metrics for Host and Avatars
    for each policy key.

    Parameters:
        file_path (str): The path to the torch-saved file containing the data.
    """
    # # Load the dictionary from the file.
    # data = torch.load(file_path)

    # Iterate over each policy key.
    for policy_key, episodes in data.items():
        print(f"\nPolicy: {policy_key}")
        
        # Containers to accumulate metrics for Host and Avatars.
        host_metrics = {}
        avatar_metrics = {}

        # Loop over each episode.
        for ep in episodes:
            # Process Host metrics.
            host_data = ep.get("Host", {})
            for metric, value in host_data.items():
                try:
                    val = float(value)
                except Exception:
                    val = np.nan
                host_metrics.setdefault(metric, []).append(val)

            # Process Avatar metrics.
            # We assume any key starting with "Avatar" is an avatar entry.
            for key, metrics in ep.items():
                if key.startswith("Avatar"):
                    for metric, value in metrics.items():
                        try:
                            val = float(value)
                        except Exception:
                            val = np.nan
                        avatar_metrics.setdefault(metric, []).append(val)

        # Compute the average metrics using np.nanmean (ignoring NaN values).
        host_avg = {
            metric: np.nan if np.all(np.isnan(filtered_values := np.array(values)[np.array(values) > 0])) 
            else np.nanmean(filtered_values)
            for metric, values in host_metrics.items()
        }

        avatar_avg = {
            metric: np.nan if np.all(np.isnan(filtered_values := np.array(values)[np.array(values) > 0])) 
            else np.nanmean(filtered_values)
            for metric, values in avatar_metrics.items()
        }

        # Print averages.
        print("  Host Average Metrics:")
        for metric, avg in host_avg.items():
            print(f"    {metric}: {avg}")

        print("  Avatars Average Metrics (aggregated):")
        for metric, avg in avatar_avg.items():
            print(f"    {metric}: {avg}")







def collect_stats(env,policy,rep_length,episod_length=60,hpolicy=None):
    obs = env.reset()
    step = 0
    agents = env.agents
    rwds = { key: []  for key in agents }
    if hpolicy != None:
        while (step < episod_length):
                    obs_,action_mask,roles = utils.process_obs(obs,rep_length=rep_length)
                    hpadded,hobs_mask = pad_mask.pad_and_mask([obs_[agent] for agent in obs_.keys() if 'Host' in agent])
                    haction_mask_ = pad_mask.pad_action_([action_mask[agent] for agent in obs_.keys() if 'Host' in agent])

                    # For the participants
                    padded,obs_mask = pad_mask.pad_and_mask([obs_[agent] for agent in obs_.keys() if 'Host' not in agent])
                    action_mask_ = pad_mask.pad_action_([action_mask[agent] for agent in obs_.keys() if 'Host' not in agent])
                    with torch.no_grad():
                        
                        hactions, hreports, _,_ = hpolicy(roles = None,obs = hpadded,obs_mask = hobs_mask ,action_mask = haction_mask_,evaluate=True)
                        actions, reports, _,_ = policy(roles = None,obs = padded,obs_mask = obs_mask ,action_mask = action_mask_,evaluate=True)
                    _, hactions_tosend = utils.actions_tosend_(hactions,hreports,{agent :action_mask[agent] for agent in obs_.keys() if 'Host' in agent })
                    _, actions_tosend = utils.actions_tosend_(actions,reports,{agent :action_mask[agent] for agent in obs_.keys() if 'Host' not in agent })

                    actions_tosend.update(hactions_tosend)

                    
                    next_obs,rewards, *_  = env.step(actions_tosend)
                    obs = next_obs
                    done = 0 if step < (episod_length-1) else 1
                    step+=1
                    obs = next_obs
                    for agent in agents:
                        rwds[agent].append(rewards[agent])
                    if done == 1:
                        stats = env._side_channel_dict["StatsSideChannel"].get_and_reset_stats()
    else : 
        while (step < episod_length):
                    obs_,action_mask,roles = utils.process_obs(obs,rep_length=rep_length)
                    padded,obs_mask = pad_mask.pad_and_mask([obs_[agent] for agent in obs_.keys()])
                    action_mask_ = pad_mask.pad_action_([action_mask[agent] for agent in obs_.keys()])
                    with torch.no_grad():
                        roles = torch.tensor(roles,dtype=torch.int32)
                        actions, reports, _,_ = policy(roles = roles,obs = padded,obs_mask = obs_mask ,action_mask = action_mask_,evaluate=True)

                    _, actions_tosend = utils.actions_tosend_(actions,reports,action_mask)
                    next_obs,rewards, *_  = env.step(actions_tosend)
                    obs = next_obs
                    done = 0 if step < (episod_length-1) else 1
                    step+=1
                    obs = next_obs
                    for agent in agents:
                        rwds[agent].append(rewards[agent])
                    if done == 1:
                        stats = env._side_channel_dict["StatsSideChannel"].get_and_reset_stats()
    rwd = {key : np.mean(rwds[key]) for key in agents}
    return rwd,stats

def compute_stats(stats):
        agents = ["Host", "Avatar_0", "Avatar_1", "Avatar_2", "Avatar_3", "Avatar_4"]
        metrics = dict()
        for agent in agents:
            # Extract TP, TN, FN, FP values from the stats dictionary
            TP = np.array([stats[agent + "_TP"][j][0] for j in range(len(stats[agent + "_TP"]))]) 
            TN = np.array([stats[agent + "_TN"][j][0] for j in range(len(stats[agent + "_TN"]))])
            FN = np.array([stats[agent + "_FN"][j][0] for j in range(len(stats[agent + "_FN"]))])
            FP = np.array([stats[agent + "_FP"][j][0] for j in range(len(stats[agent + "_FP"]))])

            # Sum only positive values greater than 0
            TP_sum = np.sum(TP[TP > 0])
            TN_sum = np.sum(TN[TN > 0])
            FN_sum = np.sum(FN[FN > 0])
            FP_sum = np.sum(FP[FP > 0])

            # You can now compute other metrics (e.g., accuracy, precision, recall, etc.) based on these sums
            total = TP_sum + TN_sum + FP_sum + FN_sum

            accuracy = (TP_sum + TN_sum) / total if total > 0 else 1
            precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 1
            recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 1
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 1
            specificity = TN_sum / (TN_sum + FP_sum) if (TN_sum + FP_sum) > 0 else 1
            metrics[agent] = {"Accuracy": accuracy, "Precision":precision,"Recall":recall,
                              "F1 Score":f1_score,"Specificity":specificity}
        return metrics
