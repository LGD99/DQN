import numpy as np
import gym
import tkinter
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_ubyte
import random
import datetime
from dqn import *
import time


class Replay_Memory:

    def __init__(self, memory_size, how_many_times_the_game_can_terminate, game):

        self.N = memory_size  # replay memory size
        self.M = how_many_times_the_game_can_terminate
        self.n = 0  # the actual size of the memory

        self.observations = []  # only 4 images from the game that are saved as a state
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states = []  # two different states have to be saved in the experience
        self.experiences = []

        self.env = gym.make(game)  # iniciazlize the environment
        self.env.reset()
        self.number_of_actions = self.env.action_space.n  # number of possible actions

        self.number_of_terminated_games = 0

        self.game_terminated = False
        self.score = 0
        self.sum_score = 0
        self.image = []

        self.save_q = False
        self.update_q = False
        self.loss = 0
        self.avgscore = 0
        self.tmp_avgscore = 0

        self.eps = EPS_START

        self.frozen_Q = DQN(self.number_of_actions).to(device)  # inicialize frozen Q
        self.net_Q = DQN(self.number_of_actions).to(device)  # inicialize Q
        self.frozen_Q.load_state_dict(self.net_Q.state_dict())

        self.optimizer = optim.Adam(self.net_Q.parameters(), lr=LEARNING_RATE)

        self.file = open('timing', 'w')
        self.opt = open('optimizer_timing', 'w')
        self.tr = open('train', 'w')
        self.st = open('10ksteps', 'w')

    def close_env(self):
        self.env.close()

    def reset_env(self):
        self.env.reset()

    def observations_append(self, new):
        self.observations.append(new)

    def states_append(self):
        tmp = []
        tmp.append(self.observations[0])
        tmp.append(self.observations[1])
        tmp.append(self.observations[2])
        tmp.append(self.observations[3])
        if len(self.states) < 2:
            self.states.append(tmp)
        else:
            self.states.pop(0)
            self.states.append(tmp)

    def save_actions(self, action):
        # action = float(action)
        if len(self.actions) < 5:
            self.actions.append(action)
        else:
            self.actions.pop(0)
            self.actions.append(action)

    def save_rewards(self):
        if self.score > 0:
            self.score = 1
        if self.score < 0:
            self.score = -1
        if len(self.rewards) < 5:
            self.rewards.append(self.score)
        else:
            self.rewards.pop(0)
            self.rewards.append(self.score)

    def save_dones(self):
        if len(self.dones) < 5:
            self.dones.append(self.game_terminated)
        else:
            self.dones.pop(0)
            self.dones.append(self.game_terminated)

    def save_experience(self):
        exp = [self.states[0], self.states[1], self.actions[0], self.rewards[0], self.dones[0]]
        if len(self.experiences) < self.N:  # N is the size of the replay memory
            self.experiences.append(exp)
        else:
            self.experiences.pop(0)
            self.experiences.append(exp)

    def preprocessing(self):
        #image = self.image
        # # image = np.array(self.env.env.ale.getScreenRGB())  #env.step
        # # plt.imshow(image)
        # image = image[31:193, 8:152, :]
        # # plt.imshow(image)
        # image = rgb2gray(image)
        # # plt.imshow(image, cmap="gray")
        # # resize
        # image = rescale(image, (84.0 / 162.0, 84.0 / 144.0), mode='constant', multichannel=False, anti_aliasing=False)
        # # plt.imshow(image, cmap="gray")
        # image = img_as_ubyte(image)

        image = resize(rgb2gray(self.image), (110, 84))[16:110 - 10, :]
        image = img_as_ubyte(image)

        if len(self.observations) < 4:
            self.observations_append(image)
        else:
            self.observations.pop(0)
            self.observations_append(image)

    def get_first_observation(self):
        for x in range(5):
            self.preprocessing()

    def save_finished_game(self):
        for x in range(4):
            self.preprocessing()
            self.save_rewards()
            self.save_dones()
            self.actions.append(self.actions[4])
            self.actions.pop(0)
            self.states_append()
            self.save_experience()
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.observations = []


    def choose_action(self):
        # compute new eps
        # linear function
        if self.eps > EPS_STOP:
            self.eps = EPS_START + (EPS_STOP - EPS_START) * self.n / EPS_FINAL_FRAME

        if len(self.experiences) < START_TO_TRAIN:
            # action = np.random.random_integers(0, self.number_of_actions - 1)
            action = torch.tensor([random.randrange(self.number_of_actions)], device=device, dtype=torch.int32)
            return action.item()

        rand = np.random.random()
        # print(rand)
        if rand < self.eps:
            action = torch.tensor([random.randrange(self.number_of_actions)], device=device, dtype=torch.int32)
            action = action.item()
        else:
            # action = np.random.random_integers(0, self.number_of_actions-1)
            # DQN choose the action
            if len(self.states) == 1:
                with torch.no_grad():
                    action = self.net_Q(torch.from_numpy(np.array([self.states[0]], dtype=np.float32)).to(device)).max(1)[1]
                action = action.item()
            if len(self.states) == 2:
                with torch.no_grad():
                    action = self.net_Q(torch.from_numpy(np.array([self.states[1]], dtype=np.float32)).to(device)).max(1)[1]
                action = action.item()
            else:
                action = torch.tensor([random.randrange(self.number_of_actions)], device=device, dtype=torch.int32)
                action = action.item()
                
        return action

    def sample(self, batch_size):
        return random.sample(self.experiences, batch_size)

    def train(self):
        self.loss, self.tmp_avgscore = optimize(self)

        if self.n % TARGET_NETWORK_UPDATE_FREQUENCY == 0:  # Update frozen Q
            self.frozen_Q.load_state_dict(self.net_Q.state_dict())
            self.update_q = True

        if self.n % 150000 == 0:
            file_name = "saved_model_%s.pt" % self.n
            torch.save(self.net_Q.state_dict(), file_name)
            self.save_q = True

    def main(self):

        f = open('statistics', 'w')
        i = 0
        k = 10000000
        stop = False

        while stop is not True:

            self.image = self.env.reset()
            #self.get_first_observation()

            while self.game_terminated is not True:
                # choose action
                action = self.choose_action()

                if RENDER == True:
                    self.env.render()

                self.image, self.score, self.game_terminated, _ = self.env.step(action)
                self.save_actions(action)
                self.save_rewards()
                self.save_dones()
                self.preprocessing()

                self.sum_score += self.score

                if len(self.observations) >= 4:
                    self.states_append()

                if self.n >= 3 & len(self.states) == 2:
                    self.save_experience()

                if len(self.experiences) > START_TO_TRAIN and (self.n % 4 == 0):
                    print(self.n, ";", file=self.tr)
                    self.train()

                if self.n % 10000 == 0 and self.n <= 50000:
                    print(datetime.datetime.now())
                    print(self.n)
                    print("1k step;", datetime.datetime.now(), file=self.st)

                if self.n % 1000 == 0 and self.n > 50000:
                    print(datetime.datetime.now())
                    print(self.n)
                    print("1k step;", datetime.datetime.now(), file=self.st)
                if self.n > 50000:
                    print(self.n, ";", self.eps, ";", self.update_q, ";", self.save_q, ";", self.tmp_avgscore, ";",
                          self.avgscore, ";", action, ";", self.score, ";", self.game_terminated, ";", self.sum_score,
                          ";", self.loss, ";", file=f)  # print(self.n, ";", self.sum_score, ";", ";\n", file=f)
                if self.n % 1000000 == 0:
                    f.close()
                    file_name = "statistics_%s" % self.n
                    f = open(file_name, 'w')
                self.save_q = False
                self.update_q = False
                self.n = self.n + 1

                # if self.game_terminated == True:
                #     i = 1
                #     k = self.n + 2
                # if self.n > k + 5 and i == 1:
                #     z = self.n -10
                #     while z < self.n+100:
                #         plt.imshow(self.experiences[z][0][0], cmap="gray")
                #         plt.imshow(self.experiences[z][0][1], cmap="gray")
                #         plt.imshow(self.experiences[z][0][2], cmap="gray")
                #         plt.imshow(self.experiences[z][0][3], cmap="gray")
                #         plt.imshow(self.experiences[z][1][0], cmap="gray")
                #         plt.imshow(self.experiences[z][1][1], cmap="gray")
                #         plt.imshow(self.experiences[z][1][2], cmap="gray")
                #         plt.imshow(self.experiences[z][1][3], cmap="gray")
                #         z += 1

            self.save_finished_game()

            self.number_of_terminated_games += 1
            self.avgscore = ((
                                         self.number_of_terminated_games - 1) / self.number_of_terminated_games) * self.avgscore + (
                                    1 / self.number_of_terminated_games) * self.sum_score
            self.game_terminated = False
            self.sum_score = 0
            if self.n > 7001000 or self.number_of_terminated_games > NUM_OF_TERMINATES:
                stop = True
        f.close()


rm = Replay_Memory(memory_size=MEMORY_SIZE, how_many_times_the_game_can_terminate=NUM_OF_TERMINATES, game=GAME)
print(datetime.datetime.now())
rm.main()
print(datetime.datetime.now())
rm.close_env()