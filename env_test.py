import gym
import numpy as np
import my_model
from PIL import Image
from env_wrapper import wrap_dqn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import my_worker

if __name__ == '__main__':

    def matrix_to_image(data):
        data1 = data * 255
        new_im = Image.fromarray(data1.astype(np.uint8))
        return new_im



    env = gym.make('PongNoFrameskip-v0')
    env = wrap_dqn(env)
    # print(env.action_space.n)

    model = my_model.CNN(4, env.action_space.n)
    # worker = my_worker.Worker(0, env, 0.05)
    # worker.this_rand_seed = 1
    # print(worker.perturbed_model.es_params())
    # worker.set_perturbed_model()
    # print("***********************")
    # print(worker.perturbed_model.es_params())

    obs = env.reset()

    for i in range(10):

        state = torch.from_numpy(obs)

        action_linear = model.forward(Variable(state.unsqueeze(0)))
        print('action_linear:', action_linear)
        prob = F.softmax(action_linear)
        print('prob:', prob)
        action = prob.max(1)[1].data.numpy()
        print('action:', action)
        obs, score, done, info = env.step(action)
        print('score:', score)
        



        # data = np.squeeze(np.split(obs, [1], axis=0)[0])
        # im = matrix_to_image(data)
        # im.save('img/'+str(i)+'.jpg')


