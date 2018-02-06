import numpy as np
import torch

class Master(object):
    def __init__(self, model, n, sigma, alpha):
        self.n = n
        self.sigma = sigma
        self.alpha = alpha
        self.weight = []
        self.max_eps_len = 10000
        self.num_generation = 0
        self.seeds = []
        self.model = model
        self.returns_to_sort = []
        
        denom = 0
        for index in range(self.n):
            denom += max(0, np.log(self.n / 2 + 1) - np.log(index + 1))
        for index in range(self.n):
            self.weight.append(
                max(0, np.log(self.n / 2 + 1) - np.log(index + 1)) / denom
                - 1 / self.n
            )

        print("master worker initialized")

    def generate_seeds(self):
        assert(len(self.seeds) == 0)
        print("seed range:")
        print(self.num_generation * self.n, self.num_generation * self.n + self.n // 2 - 1)
        for i in range(self.num_generation * self.n, self.num_generation * self.n + self.n // 2):
            self.seeds.append(self.num_generation * self.n + i)
            self.seeds.append(self.num_generation * self.n + i)

    def pull_return(self, worker):
        self.returns_to_sort.append((worker.rank, worker.this_rand_seed, worker.is_anti, worker.reward))

    def sort_returns(self):
        self.returns_to_sort.sort(key=lambda one_return: one_return[3], reverse=True)

    def update(self):
        index = 0
        for one_return in self.returns_to_sort:
            np.random.seed(one_return[1])
            for (k, v) in self.model.es_params():
                if one_return[2]:
                    z = np.random.normal(0, 1, v.size())
                    v += torch.from_numpy(self.weight[index] * z).float() * self.alpha / (self.n * self.sigma)
                else:
                    z = np.random.normal(0, 1, v.size())
                    v -= torch.from_numpy(self.weight[index] * z).float() * self.alpha / (self.n * self.sigma)
            index += 1
        self.num_generation += 1

    def reset(self):
        self.seeds = []
        self.returns_to_sort = []









