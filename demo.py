import json
import os
import matplotlib.pyplot as plt
from metaworld.benchmarks import ML1

N = 120

env_name = 'push-v1'
algo_type = 'grp'
exp_name = 'AAAA'
# env_name = 'Hopper-v2'

project_path = os.path.abspath(os.path.dirname(__file__))
result_path = project_path + '/results/'

record_path = result_path + algo_type + '/' + env_name + '/' + exp_name + '/logs/'

actions =[]
dist = 100.0
for i in range(1,150):
    print("file ",i)
    load_path = record_path + str(i).zfill(6)+'.json'
    with open(load_path) as jf:
        data =json.load(jf)
        n = data['n_scenario']
        for j in range(n):
            actions.append(data['scenario'][j]['action'])

    env = ML1.get_train_tasks(env_name)  # Create an environment with task `pick_place`
    tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
    env.set_task(tasks[0])  # Set task

    for k in range(n):
        env.reset()
        rewards = 0.0
        
        for action in actions[k]:
            state, reward, done, info = env.step(action)
            rewards += reward
            if info['distance'] < dist:
                print(info['distance'])
                dist = info['distance']
            # dist = min(info['distance'], dist)
        # print(dist)