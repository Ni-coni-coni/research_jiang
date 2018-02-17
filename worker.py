from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import copy
import gym
import time
import csv
from env_wrapper import wrap_dqn


class WorkerProcess(mp.Process):
    def __init__(self, n, rank, sigma, q_model, q_eval, max_num_generation, env_name):
        self.q_model = q_model
        self.q_eval = q_eval
        self.n = n
        self.max_num_generation = max_num_generation
        self.rank = rank
        self.env_name = env_name
        if env_name == 'CartPole-v0':
            self.env = gym.make(env_name)
        else:
            self.env = wrap_dqn(gym.make(env_name))
        self.sigma = sigma
        self.max_eps_len = 40000

        self.model = None
        self.score = 0
        self.reward = 0
        self.frame_this_eps = 0
        super(WorkerProcess, self).__init__()
        print("worker %d initialized" % rank)

    def run(self):
        for _ in range(self.max_num_generation):
            self.model = self.q_model.get()
            if self.env_name == 'CartPole-v0':
                state = self.env.reset()
                state = torch.from_numpy(state).float()
                for step in range(self.max_eps_len):
                    output = self.model(Variable(state))
                    prob = F.softmax(output)
                    action = prob.max(1)[1].data.numpy()[0][0]
                    state, score, done, _ = self.env.step(action)
                    self.score += score
                    reward = min(max(score, -1), 1)
                    self.reward += reward
                    self.frame_this_eps += 1
                    if done:
                        break
                    state = torch.from_numpy(state).float()
                # print("worker %d episode end, score %f, frames %d" % (self.rank, self.score, self.frame_this_eps))
            else:
                start = time.time()
                state = self.env.reset()
                state = torch.from_numpy(state)
                forward_accum = 0.0
                for step in range(self.max_eps_len):
                    forward_start = time.time()
                    output = self.model(Variable(state.unsqueeze(0)))
                    forward_accum += time.time() - forward_start
                    prob = F.softmax(output)
                    action = prob.max(1)[1].data.numpy()[0]
                    state, score, done, _ = self.env.step(action)
                    self.score += score
                    reward = min(max(score, -1), 1)
                    self.reward += reward
                    self.frame_this_eps += 1
                    if done:
                        break
                    state = torch.from_numpy(state)
                # elapsed = time.time() - start
                # print("worker %d episode end, score %f, frames %d, time %f sec, forward %f sec" %
                #       (self.rank, self.score, self.frame_this_eps, elapsed, forward_accum))
                # with open("csv/" + self.env_name + "_n" + str(self.n) + ".csv", "a") as csvfile:
                #     writer = csv.writer(csvfile, lineterminator='\n')
                #     writer.writerow([self.rank, self.score, self.frame_this_eps, elapsed, forward_accum])
                # csvfile.close()
            self.q_eval.put((self.rank, self.reward, self.frame_this_eps))
            self.reset()

    def reset(self):
        self.model = None
        self.score = 0
        self.reward = 0
        self.frame_this_eps = 0









