from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from parameters import *

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


def optimize(myexperiences, policy_net, target_net, optimizer):

    if len(myexperiences) < BATCH_SIZE:
        return

    experiences = myexperiences
    batch = Experience(*zip(*experiences))

     # Compute a mask of non-final states and concatenate the batch elements
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], device=device, dtype=torch.float).to(device)
    state_batch = torch.tensor(batch.state, device=device, dtype=torch.float).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long).to(device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state = target_net(non_final_next_states).max(1)
    next_state_values = next_state[0].detach()
    next_state_indexes = next_state[1].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

