from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from parameters import *
import numpy as np

class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        #Conv Layer1
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)  # (84 - (8 - 1) - 1) / 4 + 1 = 20 Output: 32 * 20 * 20
        # Conv Layer2
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # (20 - (4 -1) -1) / 2 + 1 = 9 Output: 64 * 9 * 9
        # Conv Layer3
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) # (9 - (3-1) -1)/1 + 1 = 7 Output: 64 * 7 * 7 =  3136

        self.fc1 = nn.Linear(3136, 512) #Fully Connected Layer 1
        self.fc2 = nn.Linear(512, num_actions) #Fully Connected Layer 2

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience',
                        ('state', 'next_state', 'action', 'reward', 'done'))


def optimize(myenv):

    if len(myenv.experiences) < START_TO_TRAIN:
        return 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Experience = namedtuple('Experience',
                            ('state', 'next_state', 'action', 'reward', 'done'))

    # start_time = time.time()
    experiences = myenv.sample(BATCH_SIZE)
    batch = Experience(*zip(*experiences))

    # Compute a mask of non-final states and concatenate the batch elements
    next_state_batch = torch.tensor(np.array(batch.next_state, dtype=np.float32), device=device,
                                    dtype=torch.float32).to(device)  # requires grad false
    # next_state_batch = next_state_batch / 255.
    state_batch = torch.tensor(np.array(batch.state, dtype=np.float32), device=device, dtype=torch.float32).to(
        device)  # requires grad false
    # state_batch = state_batch / 255.
    action_batch = torch.tensor(batch.action, dtype=torch.long).to(device).unsqueeze(1)  # requires grad false
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)  # requires grad false
    done_list = [not i for i in batch.done]
    done_batch = torch.tensor(np.multiply(done_list, 1), dtype=torch.float32).to(device)  # requires grad false

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = myenv.net_Q(state_batch).gather(1, action_batch)  # requires grad true
    avg_qscore = torch.sum(state_action_values.detach()) / BATCH_SIZE  # requires grad false

    # Compute max Q(s_{t+1},a') for all next states.
    next_state_action_values = myenv.frozen_Q(next_state_batch).max(1)  # requires grad true
    next_state_values = next_state_action_values[0].detach()  # requires grad false

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * done_batch * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)  # requires grad true

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values)  # grad_fn object at -> state_action_values grad_fn

    # Optimize the model
    myenv.optimizer.zero_grad()
    loss.backward()
    for param in myenv.net_Q.parameters():
        param.grad.data.clamp_(-2, 2)  # 32*4*8*8
    myenv.optimizer.step()
    # print("Optimize: %s" %(time.time()-start_time))
    return loss.item(), avg_qscore.item()