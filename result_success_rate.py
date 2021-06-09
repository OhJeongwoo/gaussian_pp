import json
import os
import matplotlib.pyplot as plt

N = 1839

algo_type = 'grp'
#algo_type = 'ppo'
env_name = 'push-v1'
exp_name = 'AABA'
# exp_name = 'imitation'

project_path = os.path.abspath(os.path.dirname(__file__))
result_path = project_path + '/results/'

record_path = result_path + algo_type + '/' + env_name + '/' + exp_name + '/logs/'

attempts = []
success = []
timestamp = []
for i in range(1,N+1):
    load_path = record_path + str(i).zfill(6) + ".json"
    with open(load_path) as jf:
        data = json.load(jf)
        n = data['n_scenario']
        attempts.append(n)
        timestamp.append(data['timestamp'])
        s = 0
        for j in range(n):
            s += data['scenario'][j]['success']
        success.append(s)

M = 5
x = timestamp[M-1:N]
y = [sum(success[i:i+M])/sum(attempts[i:i+M]) for i in range(N-M+1)]

plt.plot(x,y)
plt.show()
