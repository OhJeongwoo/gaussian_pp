import argparse
import sys
import math
from collections import namedtuple
from itertools import count
import os
import json

import gym
import numpy as np
import scipy.optimize
from gym import wrappers

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable

from models import Policy, Value, ActorCritic
from replay_memory import Memory
from running_state import ZFilter

import metaworld
from metaworld.benchmarks import ML1

# from utils import *

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
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='batch size (default: 5000)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--entropy-coeff', type=float, default=0.0, metavar='N',
                    help='coefficient for entropy cost')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='Clipping for PPO grad')
parser.add_argument('--use-joint-pol-val', action='store_true',
                    help='whether to use combined policy and value nets')
parser.add_argument('--max-ep-len', type=int, default=1e2, metavar='N',
                    help='maximum length of episode')
parser.add_argument('--algo-type', default='ppo', metavar='N')
parser.add_argument('--exp-name', type=str, default='imitation', metavar='N')

args = parser.parse_args()

#env = gym.make(args.env_name)
# env_name = "pick-place-v1"
env = ML1.get_train_tasks(args.env_name)  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task
# env = ML1.get_train_tasks(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print(args)
env.seed(args.seed)
torch.manual_seed(args.seed)

if args.use_joint_pol_val:
    ac_net = ActorCritic(num_inputs, num_actions)
    opt_ac = optim.Adam(ac_net.parameters(), lr=0.001)
else:
    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)
    opt_policy = optim.Adam(policy_net.parameters(), lr=0.001)
    opt_value = optim.Adam(value_net.parameters(), lr=0.001)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_actor_critic(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std, v = ac_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * Variable(PI)) - log_std
    return log_density.sum(1)

def update_params_actor_critic(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    action_means, action_log_stds, action_stds, values = ac_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

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

    # kloldnew = policy_net.kl_old_new() # oldpi.pd.kl(pi.pd)
    # ent = policy_net.entropy() #pi.pd.entropy()
    # meankl = torch.reduce_mean(kloldnew)
    # meanent = torch.reduce_mean(ent)
    # pol_entpen = (-args.entropy_coeff) * meanent

    action_var = Variable(actions)
    # compute probs from actions above
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old, values_old = ac_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    ac_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_ac.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()

    vf_loss1 = (values - targets).pow(2.)
    vpredclipped = values_old + torch.clamp(values - values_old, -args.clip_epsilon, args.clip_epsilon)
    vf_loss2 = (vpredclipped - targets).pow(2.)
    vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

    total_loss = policy_surr + vf_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm(ac_net.parameters(), 40)
    opt_ac.step()


def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

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

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    log_prob_cur = normal_log_density(action_var, action_means, action_log_stds, action_stds)

    action_means_old, action_log_stds_old, action_stds_old = policy_net(Variable(states), old=True)
    log_prob_old = normal_log_density(action_var, action_means_old, action_log_stds_old, action_stds_old)

    # backup params after computing probs but before updating new params
    policy_net.backup()

    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages_var = Variable(advantages)

    opt_policy.zero_grad()
    ratio = torch.exp(log_prob_cur - log_prob_old) # pnew / pold
    surr1 = ratio * advantages_var[:,0]
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages_var[:,0]
    policy_surr = -torch.min(surr1, surr2).mean()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    opt_policy.step()

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

    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        record = {}
        state_array = []
        action_array = []
        reward_array = []
        
        reward_sum = 0
        t = 0
        success = False

        while(True): # Don't infinite loop while learning
            t += 1
            if args.use_joint_pol_val:
                action = select_action_actor_critic(state)
            else:
                action = select_action(state)
            action = action.data[0].numpy()
            next_state, reward, done, info = env.step(action)
            state_array.append(state.tolist())
            action_array.append(action.tolist())
            reward_array.append(reward)
            reward_sum += reward

            next_state = running_state(next_state)

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break
            if t > args.max_ep_len:
                break
            if info['success'] > 0.0:
                success = True

            state = next_state
        num_steps += t
        num_episodes += 1
        reward_batch += reward_sum
        record['state'] = state_array
        record['action'] = action_array
        record['reward'] = reward_array
        if success:
            record['success'] = 1.0
        else :
            record['success'] = 0.0
        record['len'] = t
        scenario.append(record)
    total_steps += num_steps
    records['scenario'] = scenario
    records['n_scenario'] = len(scenario)
    records['timestamp'] = total_steps
    
    reward_batch /= num_episodes
    batch = memory.sample()
    if args.use_joint_pol_val:
        update_params_actor_critic(batch)
    else:
        update_params(batch)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}\tTotal steps {}'.format(
            i_episode, reward_sum, reward_batch, total_steps))
        record = {'reward': reward_batch, 'n_episode': i_episode, 'timestamp': total_steps, 'states': state_array, 'actions': action_array, 'rewards': reward_array}
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

