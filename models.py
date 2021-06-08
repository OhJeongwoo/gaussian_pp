import copy
import numpy as np
from numpy import linalg as LA

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from GPR import GPR

def square(a):
    return torch.pow(a, 2.)

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden=64):
        super(ActorCritic, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden)
        self.affine2 = nn.Linear(hidden, hidden)

        self.action_mean = nn.Linear(hidden, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.value_head = nn.Linear(hidden, 1)

        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std, self.value_head]
        self.module_list_old = [None]*len(self.module_list_current)
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            action_mean = self.module_list_old[2](x)
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)

            value = self.module_list_old[4](x)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))

            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

            value = self.value_head(x)

        return action_mean, action_log_std, action_std, value

class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.action_mean, self.action_log_std]

        self.module_list_old = [None]*len(self.module_list_current) #self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        # print (type(p_mean), type(p_std), type(q_mean), type(q_std))
        # q_mean = Variable(torch.DoubleTensor([q_mean])).expand_as(p_mean)
        # q_std = Variable(torch.DoubleTensor([q_std])).expand_as(p_std)
        numerator = square(p_mean - q_mean) + \
            square(p_std) - square(q_std) #.expand_as(p_std)
        denominator = 2. * square(q_std) + eps
        return torch.sum(numerator / denominator + torch.log(q_std) - torch.log(p_std))

    def kl_old_new(self):
        """Gives kld from old params to new params"""
        kl_div = self.kl_div_p_q(self.module_list_old[-2], self.module_list_old[-1], self.action_mean, self.action_log_std)
        return kl_div

    def entropy(self):
        """Gives entropy of current defined prob dist"""
        ent = torch.sum(self.action_log_std + .5 * torch.log(2.0 * np.pi * np.e))
        return ent

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            action_mean = self.module_list_old[2](x)
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))

            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


class GRPNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, n_anchors, T, dt, eps_runup):
        super(GRPNet, self).__init__()
        self.D = state_dim
        self.A = action_dim
        self.H = hidden_size
        self.N = n_anchors
        self.T = T
        self.affine1 = nn.Linear(self.D, self.H)
        self.affine2 = nn.Linear(self.H, self.H)
        self.times = nn.Linear(self.H, self.N)
        self.anchors = nn.Linear(self.H, self.N*self.A)
        self.dt = dt
        self.eps = 0.5 * dt
        self.time_array = torch.DoubleTensor([i * self.dt for i in range(1, self.T+1)]).unsqueeze(1)

        self.module_list_current = [self.affine1, self.affine2, self.times, self.anchors]

        self.module_list_old = [None]*len(self.module_list_current) #self.affine1_old, self.affine2_old, self.action_mean_old, self.action_log_std_old]
        
        #self.gpr = GPR()
        self.eps_runup = eps_runup

        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i])

    def solve(self, times, anchors, a, da, is_runup):
        if is_runup:
            times = torch.cat((times, torch.FloatTensor([0, -self.eps]).unsqueeze(1)))
            anchors = torch.cat((anchors, torch.FloatTensor([a, a- self.eps * da]).unsqueeze(1)))
        gpr = GPR()
        gpr.set_hyperparameter(0.1,0.1,0.1)
        gpr.load_data(times, anchors)
        gpr.optimize()
        mu, cov = gpr.predict_posterior(self.time_array, times, anchors)

        return mu, cov

    def forward(self, x, a, da, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))

            times = self.module_list_old[2](x)
            anchors = torch.reshape(self.module_list_old[3](x), (-1, self.A, self.N))

            E = x.shape[0]
            action_mu = []
            action_cov = []
            for e in range(E):
                for i in range(self.A):
                    if self.eps_runup and LA.norm(a[e]) > 1e-3 and LA.norm(da[e]) > 1e-3:
                        mu, cov = self.solve(times[e].unsqueeze(1), anchors[e][i].unsqueeze(1), a[e][i], da[e][i], True)
                    else :
                        mu, cov = self.solve(times[e].unsqueeze(1), anchors[e][i].unsqueeze(1), a[e][i], da[e][i], False)
                    action_mu.append(mu)
                    action_cov.append(cov)

            action_mu = torch.reshape(torch.stack(action_mu), (E, self.A, self.T))
            action_cov = torch.reshape(torch.stack(action_cov), (E, self.A, self.T, self.T))

            return action_mu, action_cov

        else :
            x = F.tanh(self.module_list_current[0](x))
            x = F.tanh(self.module_list_current[1](x))

            times = self.module_list_current[2](x)
            anchors = torch.reshape(self.module_list_current[3](x), (-1, self.A, self.N))

            E = x.shape[0]
            action_mu = []
            action_cov = []
            for e in range(E):
                for i in range(self.A):
                    if self.eps_runup and LA.norm(a[e]) > 1e-3 and LA.norm(da[e]) > 1e-3:
                        mu, cov = self.solve(times[e].unsqueeze(1), anchors[e][i].unsqueeze(1), a[e][i], da[e][i], True)
                    else :
                        mu, cov = self.solve(times[e].unsqueeze(1), anchors[e][i].unsqueeze(1), a[e][i], da[e][i], False)
                    action_mu.append(mu)
                    action_cov.append(cov)

            action_mu = torch.reshape(torch.stack(action_mu), (E, self.A, self.T))
            action_cov = torch.reshape(torch.stack(action_cov), (E, self.A, self.T, self.T))

            return action_mu, action_cov
