import numpy as np
import gym
import tkinter
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.color import rgb2gray
import random
import datetime
from dqn import *

class Replay_Memory:

    def __init__(self, memory_size, how_many_times_the_game_can_terminate, game):

        self.N = memory_size   # replay memory size
        self.M = how_many_times_the_game_can_terminate
        self.n = 0  #the actual size of the memory

        self.observations = [] #only 4 images from the game that are saved as a state
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states = [] #two different states have to be saved in the experience
        self.experiences = []

        self.env = gym.make(game) # iniciazlize the environment
        self.env.reset()
        self.number_of_actions = self.env.action_space.n #number of possible actions

        self.number_of_terminated_games = 0

        self.game_terminated = False
        self.score = 0
        self.sum_score = 0
        self.image = []

        self.eps = EPS_START

        self.frozen_Q = DQN(self.number_of_actions).to(device) # inicialize frozen Q
        self.net_Q = DQN(self.number_of_actions).to(device) # inicialize Q
        self.frozen_Q.load_state_dict(self.net_Q.state_dict())

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
        #action = float(action)
        if self.n < 5:
            self.actions.append(action)
        else:
            self.actions.pop(0)
            self.actions.append(action)

    def save_rewards(self):
        if self.score > 0:
            self.score = 1
        if self.score < 0:
            self.score = -1

        if self.n < 5:
            self.rewards.append(self.score)
        else:
            self.rewards.pop(0)
            self.rewards.append(self.score)

    def save_dones(self):
        if self.n < 5:
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
        image = self.image
        #image = np.array(self.env.env.ale.getScreenRGB())  #env.step
        #plt.imshow(image)
        image = image[31:193, 8:152, :]
        #plt.imshow(image)
        image = rgb2gray(image)
        #plt.imshow(image, cmap="gray")
        #resize
        image = rescale(image, (84.0 / 162.0, 84.0 / 144.0), mode='constant', multichannel=False, anti_aliasing=False)
        #plt.imshow(image, cmap="gray")

        if self.n < 4:
            self.observations_append(image)
        else:
            self.observations.pop(0)
            self.observations_append(image)

    def choose_action(self):
        if len(self.experiences) < START_TO_TRAIN:
            action = np.random.random_integers(0, self.number_of_actions - 1)
            return action

        rand = np.random.random()
        #print(rand)
        if rand< self.eps:
            action = np.random.random_integers(0, self.number_of_actions-1)
        else:
            #action = np.random.random_integers(0, self.number_of_actions-1)
            #DQN choose the action
            with torch.no_grad():
                action = self.net_Q(torch.from_numpy(np.array([self.states[0]], dtype=np.float32)).to(device)).max(1)[1].view(1, 1)

        #compute new eps
        #linear function
        if self.eps > EPS_STOP:
            self.eps = EPS_START + (EPS_STOP - EPS_START) * self.n / EPS_FINAL_FRAME

        return action


    def train(self, optimizer):
        optimize(random.sample(self.experiences, BATCH_SIZE), self.net_Q, self.frozen_Q, optimizer)

        if self.n % TARGET_NETWORK_UPDATE_FREQUENCY == 0: #Update frozen Q
            self.frozen_Q.load_state_dict(self.net_Q.state_dict())

        if ((self.number_of_terminated_games + 1) % (NUM_OF_TERMINATES / 15)) == 0:
            file_name = "saved_model_%s.pt" %self.n
            torch.save(self.net_Q.state_dict(), file_name)

    def main(self):
        optimizer = optim.RMSprop(self.net_Q.parameters(), lr=LEARNING_RATE)

        f = open('statistics', 'w')

        for _ in range(self.M):
            while self.game_terminated is not True:
                #choose action
                action = self.choose_action()

                if RENDER == True:
                    self.env.render()

                self.image, self.score, self.game_terminated, info = self.env.step(action)
                self.save_actions(action)
                self.save_rewards()
                self.save_dones()
                self.preprocessing()

                self.sum_score = self.sum_score + self.score

                if self.n >= 3:
                    self.states_append()

                if self.n >= 3 & len(self.states) == 2:
                    self.save_experience()

                if len(self.experiences) > START_TO_TRAIN:
                    print(self.n, ";", self.sum_score, ";", loss, ";\n", file=f)
                    self.train(optimizer)

                self.n = self.n + 1

                if self.n % 10000 == 0:
                    print(datetime.datetime.now())
                    print(self.n)

            self.env.reset()
            self.game_terminated = False
            self.number_of_terminated_games = self.number_of_terminated_games + 1
        f.close()


rm = Replay_Memory(memory_size=MEMORY_SIZE, how_many_times_the_game_can_terminate = NUM_OF_TERMINATES, game = GAME)
print(datetime.datetime.now())
rm.main()
print(datetime.datetime.now())
rm.close_env()




#írjam ki reward, fileba, iteráció szám, loss,

#eval-t csinálni

#gym monitor