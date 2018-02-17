import gym
import numpy as np
import my_model
from PIL import Image
from env_wrapper import wrap_dqn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import my_worker
import copy
import torch.multiprocessing as mp
import time

if __name__ == '__main__':

    def matrix_to_image(data):
        data1 = data * 255
        new_im = Image.fromarray(data1.astype(np.uint8))
        return new_im

    def do(model1):
        # env1 = wrap_dqn(gym.make('Pong-v0'))
        env1 = gym.make('CartPole-v0')
        state = env1.reset()
        # state = torch.from_numpy(state)
        state = torch.from_numpy(state).float()
        for step in range(1000):
            print("111")
            # output = model1(Variable(state.unsqueeze(0)))
            output = model1(Variable(state))
            print("222")
            prob = F.softmax(output)
            # action = prob.max(1)[1].data.numpy()[0]
            action = prob.max(0)[1].data.numpy()[0]
            state, score, done, _ = env1.step(action)
            if done:
                break
            # state = torch.from_numpy(state)
            state = torch.from_numpy(state).float()
        print("worker episode end, score, frames")




    # env = wrap_dqn(gym.make('Pong-v0'))
    env = gym.make('CartPole-v0')

    # print(env.action_space.n)

    model = my_model.SimpleNetwork(4, env.action_space.n)
    worker = my_worker.Worker(0, env, 0.05)

    obs = env.reset()

    for i in range(10):
        state = torch.from_numpy(obs).float()
        # print('state: ', state)
        # print(Variable(state.unsqueeze(0)))
        # action_linear = model(Variable(state.unsqueeze(0)))
        action_linear = model(Variable(state))
        # print('action_linear:', action_linear)
        prob = F.softmax(action_linear, dim=1)
        print('prob: ', prob)
        # action = prob.max(1)[1].data.numpy()[0]  # TODO cnn: prob.max(1)[1] delete[0]
        action = prob.max(0)[1].data.numpy()[0]
        print('action: ', action)
        obs, score, done, info = env.step(action)
        # print('score: ', score)
        # data = np.squeeze(np.split(obs, [1], axis=0)[0])
        # im = matrix_to_image(data)
        # im.save('img/'+str(i)+'.jpg')





