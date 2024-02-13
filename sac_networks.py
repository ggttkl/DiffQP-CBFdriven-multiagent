import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

from qpth.qp import QPFunction
from torch.autograd import Function, Variable

class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_hid1, n_hid2, lr, checkpoint_file, name):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = checkpoint_file + name

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name):
        super(CriticNetwork, self).__init__()

        input_len = n_states + n_actions
        self.fc1 = nn.Linear(input_len, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, 1)  # q1 layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = checkpoint_file + name

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # q1
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, max_action, lr, checkpoint_file, name):
        super(ActorNetwork, self).__init__()

        self.reparam_noise = 1e-6  # noise parameter for the reparametrization trick

        self.max_action = max_action
        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3_mu = nn.Linear(n_hid2, n_actions)
        self.fc3_sigma = nn.Linear(n_hid2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = checkpoint_file + name

        # QP params.
        self.nBatch = 1
        self.nZ = n_actions + 1  # 决策变量的数量
        self.nG = 2  # 不等式约束的数量
        self.Q = torch.rand(self.nBatch, self.nZ, self.nZ).to(self.device)
        self.q = torch.rand(self.nBatch, self.nZ).to(self.device)
        self.G = torch.rand(self.nBatch, self.nG, self.nZ).to(self.device)
        self.h = torch.rand(self.nBatch, self.nG).to(self.device)
        # hyper parameters
        self.Ds = torch.tensor(1.0).to(self.device)
        self.Dc = torch.tensor(0.1).to(self.device)
        self.alpha1 = torch.tensor(1.0, requires_grad=True).to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        sigma = self.fc3_sigma(x)
        # we are using 1e-6 instead of 0, because it may throw an error when sampling a distribution with a st.dev of 0.
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)  # in paper they set (-20, 1)
        return mu, sigma

    def sample_normal(self, state, reparametrize=True):

        # for the deterministic policy test just return mu instead of action
        mu, sigma = self.forward(state)  # get mu and sigma for the state
        probabilities = torch.distributions.Normal(mu, sigma)  # get distribution
        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        # # actions to QP layers
        # self.nBatch = state.size(0)
        # x_current = state[:, :2].to(self.device)
        # x_ref = state[:, 2:4].to(self.device)
        # x_obstacle = state[:, 4:].to(self.device)
        # # action = -0.0 * (x_current - x_ref)


        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)

        # action_max = 100.0
        # action_min = -100.0
        # action_transformed = (action + 1) * 0.5 * (action_max - action_min) + action_min
        #
        # h_obs = (torch.norm(x_current - x_obstacle, p=2, dim=1, keepdim=True) ** 2 - (self.Ds ** 2) * torch.ones(self.nBatch,
        #                                                                                                     1).to(
        #     self.device)).to(self.device)
        # h_task = (-torch.norm(x_current - x_ref, p=2, dim=1, keepdim=True) ** 2 + (self.Dc ** 2) * torch.ones(
        #     self.nBatch,
        #     1).to(
        #     self.device)).to(self.device)
        #
        # # ---two tasks with slack variable
        # diag_matrix = torch.eye(3)
        # self.Q = diag_matrix.repeat(self.nBatch, 1, 1).to(self.device)
        # q = -action_transformed.to(self.device)
        # zero_column = torch.zeros(self.nBatch, 1).to(self.device)  # 创建一个形状为 (256, 1) 的全零列
        # self.q = torch.cat((q, zero_column), dim=1)  # 拼接两个 tensor
        # G_obs = -(2.0 * (x_current - x_obstacle)).unsqueeze(1).to(self.device)
        # minus_one_tensor = torch.full((self.nBatch, 1, 1), -0.1).to(self.device)
        # minus_zero_tensor = torch.full((self.nBatch, 1, 1), 0.0).to(self.device)
        # G_task = -(-2.0 * (x_current - x_ref)).unsqueeze(1).to(self.device)
        # G_obs_slack = torch.cat((G_obs, minus_zero_tensor), dim=2)
        # G_task_slack = torch.cat((G_task, minus_one_tensor), dim=2)
        # # G_task_slack = torch.cat((G_task, minus_zero_tensor), dim=2)
        #
        # # self.G = torch.cat((G_task_slack, G_obs_slack), dim=1)
        # self.G = G_obs_slack
        # h_task = (5.0 * h_task).to(self.device)
        # h_obs = (50.0 * h_obs).to(self.device)
        # # self.h = torch.cat((h_task, h_obs), dim=1)
        # self.h = h_obs



        # e = Variable(torch.Tensor()).to(self.device)
        # solver_args = {"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-3}
        # # solver_args = {"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4} # standard
        # acitons_QP= QPFunction(verbose=-1, **solver_args)(self.Q, self.q, self.G, self.h, e, e).to(self.device)
        # actions_trunc = acitons_QP[:, :2]
        # # actions_trunc = acitons_QP
        # actions_reversed = 2 * (actions_trunc - action_min) / (action_max - action_min) - 1
        #
        # actions_reversed_clamped = torch.clamp(actions_reversed, min=-1, max=1)

        # handle action bounds,
        # by taking the tanh activation and multiplying it by the max_action for our environment
        # action = torch.tanh(acitons_QP) * torch.tensor(self.max_action).to(self.device)
        # action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        # log_probs -= torch.log(1-actions_reversed_clamped.pow(2) + self.reparam_noise)  # need to add reparam_noise because we cannot take the log of 0, and is possible that the square of the action is 1.
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        # if torch.isnan(log_probs).any():
        #     print(f"actions_reversed={actions_reversed},log_probs={log_probs}")

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))

