import torch
import gym
import time
import copy
import master
import worker
import torch.multiprocessing as mp
from env_wrapper import wrap_dqn

if __name__ == '__main__':
    debug = False
    env_name = 'PongNoFrameskip-v0'
    env = wrap_dqn(gym.make(env_name))
    n = 24
    sigma = 0.5
    alpha = 0.1
    seed = 100
    max_num_generation = 100
    queues_model = []
    queues_eval = []
    for rank in range(n):
        q_model = mp.Queue(maxsize=1)
        queues_model.append(q_model)
        q_eval = mp.Queue(maxsize=1)
        queues_eval.append(q_eval)

    p_workers = []
    processes = []
    p_master = master.MasterProcess(n, sigma, alpha, seed, env.action_space.n,
                                    queues_model, queues_eval, max_num_generation, env_name)

    for rank in range(n):
        p_worker = worker.WorkerProcess(n, rank, sigma, queues_model[rank], queues_eval[rank],
                                        max_num_generation, env_name)
        p_workers.append(p_worker)
        p_worker.start()

        # time.sleep(0.1)
    p_master.start()
    for p_worker in p_workers:
        p_worker.start()
        time.sleep(0.05)

    p_master.join()
    print("process ", p_master, " join")

    for p_worker in p_workers:
        time.sleep(0.1)
        p_worker.join()
        print("process ", p_worker, " join")









