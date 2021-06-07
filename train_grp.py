import argparse
import sys
import math
from collections import namedtuple
from itertools import count
import os
import json
import psutil
import time
import random

import gym
import numpy as np
from numpy import linalg as LA
import scipy.optimize
from gym import wrappers

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

from models import Policy, Value, ActorCritic, GRPNet
from replay_memory import Memory
from running_state import ZFilter
import wandb

# from metaworld.benchmarks import ML1
import metaworld

# from utils import *

start_time = time.time()

torch.set_default_tensor_type('torch.DoubleTensor')
PI = torch.DoubleTensor([3.1415926])

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
# parser.add_argument('--l2_reg', type=float, default=1e-3, metavar='G',
#                     help='l2 regularization regression (default: 1e-3)')
# parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',
#                     help='max kl value (default: 1e-2)')
# parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
#                     help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='batch size (default: 5000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.01, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--dt', type=float, default=0.01, metavar='N')
parser.add_argument('--hidden-size', type=int, default=64, metavar='N')
parser.add_argument('--N', type=int, default=10, metavar='N',
                    help='number of anchors')
parser.add_argument('--T', type=int, default=5, metavar='N',
                    help='number of predictions')
parser.add_argument('--max-ep-len', type=int, default=1e2, metavar='N',
                    help='maximum length of episode')
parser.add_argument('--algo-type', default='grp', metavar='N')
parser.add_argument('--eps-runup', type=bool, default=True, metavar='N')
parser.add_argument('--exp-name', type=str, default='001', metavar='N')
parser.add_argument('--use-entropy-loss', type=bool, default=False, metavar='N')

args = parser.parse_args()

wandb.init(project = "test", reinit = True, name = args.exp_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
#env = gym.make(args.env_name)
#env_name = "pick-place-v1"
ml1 = metaworld.ML1(args.env_name)
env = ml1.train_classes[args.env_name]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
# env = ML1.get_train_tasks(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print(args)
env.seed(args.seed)
torch.manual_seed(args.seed)

# if args.use_joint_pol_val:
#     ac_net = ActorCritic(num_inputs, num_actions)
#     opt_ac = optim.Adam(ac_net.parameters(), lr=0.001)
# else:
#     policy_net = Policy(num_inputs, num_actions)
#     value_net = Value(num_inputs)
#     opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
#     opt_value = optim.Adam(value_net.parameters(), lr=0.001)
policy_net = GRPNet(num_inputs, num_actions, args.hidden_size, args.N, args.T, args.dt, args.eps_runup, device).to(device)
value_net =  Value(num_inputs, device).to(device)
opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
opt_value = optim.Adam(value_net.parameters(), lr=0.001)

def select_actions(state, a, da):
    state =  torch.from_numpy(state).unsqueeze(0).to(device)
    a = torch.from_numpy(a).unsqueeze(0).to(device)
    da = torch.from_numpy(da).unsqueeze(0).to(device)
    action_mu, action_cov = policy_net(Variable(state), a, da)
    action_mu = action_mu.squeeze(0)
    action_cov = action_cov.squeeze(0)
    actions = []
    for i in range(num_actions):
        dist = MultivariateNormal(action_mu[i], action_cov[i])
        actions.append(dist.rsample())
    return torch.stack(actions)



# def select_action(state):
#     state = torch.from_numpy(state).unsqueeze(0)
#     action_mean, _, action_std = policy_net(Variable(state))
#     action = torch.normal(action_mean, action_std)
#     return action

# def select_action_actor_critic(state):
#     state = torch.from_numpy(state).unsqueeze(0)
#     action_mean, _, action_std, v = ac_net(Variable(state))
#     action = torch.normal(action_mean, action_std)
#     return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def multivariative_normal_log_density(x, mu, cov):
    dist = MultivariateNormal(mu, cov)
    return dist.log_prob(x).sum(1)

def multivariative_normal_entropy(mu, cov):
    dist = MultivariateNormal(mu, cov)
    return dist.entropy()

def update_params(batch):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(np.array(batch.action)).to(device)
    a = torch.Tensor(np.array(batch.a)).to(device)
    da = torch.Tensor(np.array(batch.da)).to(device)
    states = torch.Tensor(batch.state).to(device)
    values = value_net(Variable(states))
    returns = torch.Tensor(actions.size(0),1).to(device)
    deltas = torch.Tensor(actions.size(0),1).to(device)
    advantages = torch.Tensor(actions.size(0),1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    opt_value.zero_grad()
    value_loss = (values - targets).pow(2.).mean()
    value_loss.backward()
    opt_value.step()

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent
    action_var = Variable(actions)
    action_mu, action_cov = policy_net(Variable(states), a, da)
    log_prob_cur = multivariative_normal_log_density(action_var, action_mu, action_cov)
    action_mu_old, action_cov_old = policy_net(Variable(states), a, da, old=True)
    log_prob_old = multivariative_normal_log_density(action_var, action_mu_old, action_cov_old)
    # backup params after computing probs but before updating new params
    policy_net.backup()
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)
    
    opt_policy.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()
    if args.use_entropy_loss:
        entropy_list = multivariative_normal_entropy(action_mu, action_cov)
        entropy_loss = entropy_list.mean()
        policy_surr -= args.entropy_coeff * entropy_loss
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    opt_policy.step()


print("=============================================")
print("experiment name          : ", args.exp_name)
print("environment name         : ", args.env_name)
print("algorithm type           : ", args.algo_type)
print("seed number              : ", args.seed)
print("batch size               : ", args.batch_size)
print("maximum episode length   : ", args.max_ep_len)
print("gamma                    : ", args.gamma)
print("tau                      : ", args.tau)
print("clip epsilon             : ", args.clip_epsilon)
print("use entropy loss         : ", args.use_entropy_loss)
print("entropy loss coefficient : ", args.entropy_coeff)
print("use epsilon run-up       : ", args.eps_runup)
print("number of anchors        : ", args.N)
print("number of predictions    : ", args.T)
print("delta t                  : ", args.dt)
print("state dimension          : ", num_inputs)
print("action dimension         : ", num_actions)
print("hidden size              : ", args.hidden_size)
print("=============================================")

summary_path = 'results/' + args.algo_type + '/' + args.env_name + '/' + args.exp_name + '/'
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
summary = {}
summary['exp_name'] = args.exp_name
summary['env_name'] = args.env_name
summary['algo_type'] = args.algo_type
summary['seed'] = args.seed
summary['batch_size'] = args.batch_size
summary['max_ep_len'] = args.max_ep_len
summary['gamma'] = args.gamma
summary['tau'] = args.tau
summary['clip_epsilon'] = args.clip_epsilon
summary['use_entropy_loss'] = args.use_entropy_loss
summary['entropy_coeff'] = args.entropy_coeff
summary['eps_runup'] = args.eps_runup
summary['N'] = args.N
summary['T'] = args.T
summary['dt'] = args.dt
summary['hidden_size'] = args.hidden_size

        
with open(summary_path + 'summary.json', 'w') as fp:
    json.dump(summary, fp, indent=4)   
    print("Save summary file ", summary_path)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)
episode_lengths = []

total_steps = 0

for i_episode in range(1, 1001):
    memory = Memory()
    records = {}
    records['i_episode'] = i_episode
    scenario = []

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size * args.T:
        state = env.reset()
        state = running_state(state)

        a = np.array([0] * num_actions)
        da = np.array([0] * num_actions)

        record = {}
        state_array = []
        action_array = []
        reward_array = []
        
        reward_sum = 0
        t = 0
        success = False
        while(True): # Don't infinite loop while learning
            backup_state = state
            backup_a = a
            backup_da = da
            actions = select_actions(state, a, da)
            actions = actions.cpu().detach().numpy()
            rewards = 0
            
            for i in range(args.T):
                action = actions[:,i]
                next_state, reward, done, info = env.step(action)
                state_array.append(state.tolist())
                action_array.append(action.tolist())
                reward_array.append(reward)
                rewards += reward
                
                if LA.norm(a) > 1e-3: # it means a is not None
                    da = (action - a) / args.dt
                a = action

                next_state = running_state(next_state)

                t += 1
                mask = 1
                
                
                if done:
                    mask = 0
                    break


                state = next_state

            memory.push(backup_state, np.array(actions), mask, next_state, rewards, backup_a, backup_da)
            reward_sum += rewards

            if args.render:
                env.render()
            if done:
                break
            if t > args.max_ep_len:
                    break
        num_steps += t
        num_episodes += 1
        reward_batch += reward_sum
        record['state'] = state_array
        record['action'] = action_array
        record['reward'] = reward_array
        record['ep_rewards'] = reward_sum
        record['success'] = info['success']
        record['len'] = t
        scenario.append(record)
    
    total_steps += num_steps
    records['scenario'] = scenario
    records['n_scenario'] = len(scenario)
    records['timestamp'] = total_steps

    reward_batch /= num_episodes
    
    batch = memory.sample()
    update_params(batch)

    if i_episode % args.log_interval == 0:
        print('[{:.2f}] Episode {}\tLast reward: {}\tAverage reward {:.2f}\tTotal steps {}'.format(
            time.time() - start_time, i_episode, reward_sum, reward_batch, total_steps))
        wandb.log({"reward": reward_batch})
        # record = {'reward': reward_batch, 'n_episode': i_episode, 'timestamp': total_steps}
        checkdir = 'results/' + args.algo_type + '/' + args.env_name + '/' + args.exp_name + '/logs/'
        weight_path = 'results/' + args.algo_type + '/' + args.env_name + '/' + args.exp_name + '/weights/'
        if not os.path.exists(checkdir):
            os.makedirs(checkdir)
        if not os.path.exists(weight_path + 'actor/'):
            os.makedirs(weight_path + 'actor/')
        if not os.path.exists(weight_path + 'critic/'):
            os.makedirs(weight_path + 'critic/')
        
        savepath = os.path.join(checkdir, '%.6i.json'%i_episode)
        with open(savepath, 'w') as fp:
            json.dump(records, fp, indent=4)
        torch.save(policy_net.state_dict(), weight_path + 'actor/%.6i.pth' % i_episode)
        torch.save(value_net.state_dict(), weight_path + 'critic/%.6i.pth' % i_episode)
