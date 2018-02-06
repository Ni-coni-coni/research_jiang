from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import copy


class Worker(object):
    def __init__(self, rank, env, sigma):
        self.rank = rank
        self.env = env
        self.max_eps_len = 10000
        self.sigma = sigma

        self.model = None
        self.perturbed_model = None
        self.is_anti = None
        self.this_rand_seed = None

        self.score = 0
        self.reward = 0
        self.frame_this_eps = 0
        print("worker %d initialized" % rank)

    def pull_seed(self, master):
        self.this_rand_seed = master.seeds[self.rank]
        if self.rank % 2 == 0:
            self.is_anti = False
        else:
            self.is_anti = True

    def pull_model(self, master):
        self.model = copy.deepcopy(master.model)
        self.perturbed_model = copy.deepcopy(master.model)

    def set_perturbed_model(self):
        np.random.seed(self.this_rand_seed)
        if self.is_anti:
            for (k, v), (k1, v1) in zip(self.model.es_params(), self.perturbed_model.es_params()):
                z = np.random.normal(0, 1, v.size())
                torch.add(v, torch.from_numpy(self.sigma * -z).float(), out=v1)
        else:
            for (k, v), (k1, v1) in zip(self.model.es_params(), self.perturbed_model.es_params()):
                z = np.random.normal(0, 1, v.size())
                torch.add(v, torch.from_numpy(self.sigma * z).float(), out=v1)

    def do_rollout(self, max_eps_len):
        state = self.env.reset()
        state = torch.from_numpy(state)
        for step in range(max_eps_len):
            print("111")
            output = self.model(Variable(state.unsqueeze(0)))
            print("222")
            prob = F.softmax(output)
            action = prob.max(1)[1].data.numpy()
            state, score, done, _ = self.env.step(action[0])
            self.score += score
            reward = min(max(score, -1), 1)
            self.reward += reward
            self.frame_this_eps += 1
            if done:
                break
            state = torch.from_numpy(state)
        print("worker %d episode end, score %f, frames %d" % (self.rank, self.score, self.frame_this_eps))

    def reset(self):
        self.model = None
        self.perturbed_model = None
        self.is_anti = None
        self.this_rand_seed = None
        self.score = 0
        self.reward = 0
        self.frame_this_eps = 0









