import json
import os
import matplotlib.pyplot as plt

N = 120

env_name = 'push-v1'
algo_type = 'grp'
exp_name = 'AAAA'
# env_name = 'Hopper-v2'

project_path = os.path.abspath(os.path.dirname(__file__))
result_path = project_path + '/results/'

record_path = result_path + algo_type + '/' + env_name + '/' + exp_name + '/logs/'

print(project_path)
time_step = []
rewards = []
for i in range(1,N+1):
    load_path = record_path + str(i).zfill(6) + ".json"
    with open(load_path) as jf:
        data = json.load(jf)
        #time_step.append(data['timestamp'])
        #rewards.append(data['reward'])
        for j in range(data['n_scenario']):
            print(data['scenario'][j]['success'])
"""
M = 5
x = time_step[M-1:N]
y = [sum(rewards[i:i+M])/M for i in range(N-M+1)]

plt.plot(x,y)
plt.show()
"""
