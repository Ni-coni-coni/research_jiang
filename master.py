import numpy as np
import torch
import copy
import torch.multiprocessing as mp
import my_model
import time
import csv
import gym
from env_wrapper import wrap_dqn
from torch.autograd import Variable
import torch.nn.functional as F


class MasterProcess(mp.Process):
    def __init__(self, n, sigma, alpha, seed, num_action,
                 queues_model, queues_eval, max_num_generation, env_name):
        assert(n % 2 == 0)
        self.max_num_generation = max_num_generation
        self.queues_model = queues_model
        self.queues_eval = queues_eval
        self.n = n
        self.sigma = sigma
        self.alpha = alpha
        self.weights = []
        self.max_eps_len = 40000
        self.num_generation = 0
        self.seed = seed
        self.env_name = env_name
        if env_name == 'CartPole-v0':
            self.env = gym.make(env_name)
        else:
            self.env = wrap_dqn(gym.make(env_name))

        self.score = 0
        self.reward = 0
        self.frame_total = 0

        self.evals = []
        if env_name == 'CartPole-v0':
            self.model = my_model.SimpleNetwork(4, num_action)
        else:
            self.model = my_model.CNN(4, num_action)
        self.z_lists = [[] for _ in range(n // 2)]

        np.random.seed(seed)

        denom = 0
        for index in range(self.n):
            denom += max(0, np.log(self.n / 2 + 1) - np.log(index + 1))
        for index in range(self.n):
            self.weights.append(
                max(0, np.log(self.n / 2 + 1) - np.log(index + 1)) / denom - 1 / self.n
            )

        super(MasterProcess, self).__init__()

        print("master worker initialized")

    def run(self):
        for _ in range(self.max_num_generation):
            print("*************************************************************")
            print("generation: ", self.num_generation)
            start = time.time()
            for rank in range(self.n):
                if rank % 2 == 0:
                    model, anti_model = self.generate_two_models(rank)
                    self.queues_model[rank].put(model)
                    self.queues_model[rank+1].put(anti_model)
                    # self.master_conns[rank].close()
                    # self.master_conns[rank+1].close()
            state = self.env.reset()
            state = torch.from_numpy(state)
            for step in range(self.max_eps_len):
                output = self.model(Variable(state.unsqueeze(0)))
                prob = F.softmax(output)
                action = prob.max(1)[1].data.numpy()[0]
                state, score, done, _ = self.env.step(action)
                self.score += score
                reward = min(max(score, -1), 1)
                self.reward += reward
                self.frame_total += 1
                if done:
                    break
                state = torch.from_numpy(state)
            print("test episode %d end, score %f, total frames %d" %
                  (self.num_generation, self.score, self.frame_total))
            with open("csv/" + self.env_name + "_n" + str(self.n) + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile, lineterminator='\n')
                writer.writerow([self.frame_total, self.score])
            csvfile.close()

            for rank in range(self.n):
                self.evals.append(self.queues_eval[rank].get())
            self.sort_evals()
            self.update()
            self.reset()
            loop_time = time.time() - start
            print("loop_time:{0}".format(loop_time) + "[sec]")
            # print("test episode end, score %f, frames %d" % (self.score, self.frame_total))
            # with open("csv/" + self.env_name + "_n" + str(self.n) + ".csv", "a") as csvfile:
            #     writer = csv.writer(csvfile, lineterminator='\n')
            #     writer.writerow([loop_time])
            # csvfile.close()

    def generate_two_models(self, prev_rank):
        assert(prev_rank % 2 == 0)
        target_model = copy.deepcopy(self.model)
        anti_target_model = copy.deepcopy(self.model)
        for (k, v), (k1, v1), (k2, v2) in \
                zip(self.model.es_params(), target_model.es_params(), anti_target_model.es_params()):
            z = np.random.normal(0, 1, v.size())
            torch.add(v, torch.from_numpy(self.sigma * z).float(), out=v1)
            torch.add(v, torch.from_numpy(self.sigma * -z).float(), out=v2)
            self.z_lists[prev_rank//2].append(z)
        return target_model, anti_target_model

    def sort_evals(self):
        self.evals.sort(key=lambda x: x[1], reverse=True)

    def update(self):
        start = time.time()
        for ranking_idx, eval_tuple in enumerate(self.evals):
            rank = eval_tuple[0]
            l = (1 if rank % 2 == 0 else -1)
            for v_idx, (k, v) in enumerate(self.model.es_params()):
                v += torch.from_numpy(self.weights[ranking_idx] * self.z_lists[rank//2][v_idx] * l).float() * \
                    self.alpha / (self.n * self.sigma)
        self.num_generation += 1
        elapsed = time.time() - start
        print("update_time:{0}".format(elapsed) + "[sec]")
        for eval_tuple in self.evals:
            self.frame_total += eval_tuple[2]

    def reset(self):
        self.evals = []
        self.z_lists = [[] for _ in range(self.n // 2)]
        self.score = 0
        self.reward = 0









