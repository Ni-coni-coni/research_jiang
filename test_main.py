import torch.multiprocessing as mp
import gym
import my_model
import my_worker
import my_master
import time
from env_wrapper import wrap_dqn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

max_num_generation = 5
n = 4
lock = mp.Lock()


def rollout(worker1, max_eps_len):
    state = worker1.env.reset()
    state = torch.from_numpy(state)
    for step in range(max_eps_len):
        print("step ", step)
        output = worker1.model(Variable(state.unsqueeze(0)))
        print("get output ", step)
        prob = F.softmax(output)
        action = prob.max(1)[1].data.numpy()
        state, score, done, _ = worker1.env.step(action[0])
        worker1.score += score
        reward = min(max(score, -1), 1)
        worker1.reward += reward
        worker1.frame_this_eps += 1
        if done:
            break
        state = torch.from_numpy(state)
    print("worker %d episode end, score %f, frames %d" % (worker1.rank, worker1.score, worker1.frame_this_eps))

def train_loop(master):
    master.generate_seeds()
    processes = []
    max_eps_len = master.max_eps_len
    for i in range(n):
        workers[i].pull_seed(master)
        workers[i].pull_model(master)
        workers[i].set_perturbed_model()
        print("worker %d model ready" % i)
        print(id(workers[i].model))

    # rollout(workers[2], max_eps_len)

    # p = mp.Process(target=rollout, args=(workers[2], max_eps_len,))
    # p.start()
    # processes.append(p)
    # time.sleep(0.1)

    # p.join()

    for i in range(n):
        p = mp.Process(target=workers[i].do_rollout, args=(max_eps_len,))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

    for i in range(n):
        master.pull_return(workers[i])
        workers[i].reset()
    master.sort_returns()
    master.update()
    master.reset()


if __name__ == '__main__':
    env = gym.make('PongNoFrameskip-v0')
    env = wrap_dqn(env)
    model = my_model.CNN(4, env.action_space.n)
    master = my_master.Master(model, n, sigma=0.1, alpha=0.1)
    workers = []
    for rank in range(n):
        worker_env = gym.make('PongNoFrameskip-v0')
        worker_env = wrap_dqn(worker_env)
        worker = my_worker.Worker(rank, env, sigma=0.1)
        workers.append(worker)
    for _ in range(max_num_generation):
        train_loop(master)









