import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import math
import random

class AuvEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AuvEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # AUV dynamics parameters
        self.timestep = 0.01

        # Initialize rendering
        self.fig = None
        self.ax = None
        self.auv_patch = None
        self.positions = []
        self.obs_cnt = 0

        self.reset()

    def reset(self, seed=None, obs_cnt=1, agt_cnt=2):
        self.obs_cnt = obs_cnt
        self.agt_cnt = agt_cnt
        state_dim = agt_cnt*4+obs_cnt*2
        action_dim = agt_cnt*2
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.state = 5 * np.random.randn(state_dim,)
        self.ref = np.concatenate((self.state[2:agt_cnt*2], self.state[:2]))
        self.state[2*agt_cnt:4*agt_cnt] = self.ref
        # self.ref = np.concatenate(
        #     (self.state[2:state_dim - obs_cnt * 2], self.state[:2], self.state[state_dim - obs_cnt * 2:]))
        # self.ref = 5 + 5 * np.random.randn(agt_cnt*2, )
        # theta = 2 * math.pi * random.random()
        # self.ref[0] = self.state[0] + 10 * math.cos(theta)  # 10 is radius
        # self.ref[1] = self.state[1] + 10 * math.sin(theta)
        self.obstacle = 0.5 * (self.ref[0:2] + self.state[0:2]) + 0.2 * np.random.randn(2,)
        self.eps_cnt = 0
        self.max_episode_steps = 50  # 1s 100steps reach
        self.state[state_dim - obs_cnt * 2:] = self.obstacle

        self.hreach = 0
        self.hobs = 0
        self.action_max = 100
        self.action_min = -100
        self.bound = 20
        self.alpha_1 = 10
        self.alpha_2 = 100
        self.safedis = 1.0

        self.positions = [[] for _ in range(agt_cnt)]
        # self.positions = []
        self.last_goal_dist = np.linalg.norm(self.state[0:agt_cnt*2] - self.state[agt_cnt*2:state_dim - obs_cnt * 2])
        print(f"self.obs_cnt={self.obs_cnt}")
        return self.state, {}

    def step(self, action):
        self.eps_cnt += 1
        for i in range(self.agt_cnt):
            # 计算每个机器人的新位置
            action_transformed = (action[i * 2:(i + 1) * 2] + 1) * 0.5 * (
                        self.action_max - self.action_min) + self.action_min
            etadot = action_transformed
            xdot, ydot = etadot
            self.state[i * 2:(i + 1) * 2] = self.state[i * 2:(i + 1) * 2] + np.array([xdot, ydot]) * self.timestep
            # 记录轨迹
            self.positions[i].append(self.state[i * 2:(i + 1) * 2].copy())


        # self.positions.append(self.state[0:2].copy())
        # self.eps_cnt += 1
        # action_transformed = (action + 1) * 0.5 * (self.action_max - self.action_min) + self.action_min
        # # action_transformed = action
        # current_err = self.state[0:2] - self.state[2:4]
        # self.hreach = -np.linalg.norm(self.state[0:2] - self.state[2:4]) ** 2
        # etadot = action_transformed
        # xdot, ydot = etadot
        # self.state[0:2] = self.state[0:2] + np.array([xdot, ydot]) * self.timestep
        # self.positions.append(self.state[0:2].copy())

        # self.hobs = np.linalg.norm(self.state[0:2] - self.state[4:6]) ** 2 - self.safedis ** 2
        # if self.hobs < 0.0:  # action后的位置是否碰撞
        #     self.obs_cnt += 1

        done = False
        # --------redefine reward----------------
        dist_goal = np.linalg.norm(self.state[0:self.agt_cnt*2] - self.state[self.agt_cnt*2:self.agt_cnt*4])
        reward = 2 * (self.last_goal_dist - dist_goal)  # -1e-3 * dist_goal
        # reward += 0.1 * math.exp(-np.linalg.norm(self.state[0:2] - self.state[2:4]))
        self.last_goal_dist = dist_goal
        if np.linalg.norm(dist_goal) <= 1.0:
            reward += 0.5 * (self.max_episode_steps - self.eps_cnt)
            done = True

        if self.eps_cnt >= self.max_episode_steps:
            done = True

        truncated = False
        info = {}
        return self.state, reward, done, truncated, info


    def render(self, mode='human'):
        # if self.fig is None:
        #     plt.ion()  # Enable interactive mode
        #     self.fig, self.ax = plt.subplots()
        #     self.ax.set_title("AUV Position and Heading")
        #     self.ax.set_xlabel("X-axis")
        #     self.ax.set_ylabel("Y-axis")
        #     self.ax.set_xlim(-20, 20)
        #     self.ax.set_ylim(-20, 20)
        #     triangle_points = np.array([[0, -5], [-3, 5], [3, 5]])
        #     self.auv_patch = patches.Polygon(triangle_points, closed=True, color='blue', alpha=0.6)
        #     self.ax.add_patch(self.auv_patch)
        # else:
        #     # Clear the axes if it's not the first call to render
        #     self.ax.clear()
        #     # Re-set axes properties after clearing
        #     self.ax.set_xlim(-20, 20)
        #     self.ax.set_ylim(-20, 20)
        if self.fig is None:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("AUV Positions and Headings")
            self.ax.set_xlabel("X-axis")
            self.ax.set_ylabel("Y-axis")
            self.ax.set_xlim(-20, 20)
            self.ax.set_ylim(-20, 20)
            self.auv_patches = []
            for _ in range(4):
                triangle_points = np.array([[0, -5], [-3, 5], [3, 5]])
                auv_patch = patches.Polygon(triangle_points, closed=True, color='blue', alpha=0.6)
                self.auv_patches.append(auv_patch)
                self.ax.add_patch(auv_patch)
        else:
            self.ax.clear()
            self.ax.set_xlim(-20, 20)
            self.ax.set_ylim(-20, 20)

        colors = ['blue', 'red', 'green', 'yellow']
        # circle = plt.Circle(self.ref[0:2], 1, color='red', alpha=0.4)
        # self.ax.add_patch(circle)
        circle_obs = plt.Circle(self.obstacle[0:2], 1, color='gray', alpha=0.4)
        self.ax.add_patch(circle_obs)
        for i in range(self.agt_cnt):
            circle = plt.Circle(self.ref[i*2:2+i*2], 1, color=colors[i], alpha=0.4)
            self.ax.add_patch(circle)


        for i in range(self.agt_cnt):
            x, y = self.state[i * 2], self.state[i * 2 + 1]
            heading = 0  # 如果有航向信息，这里应该更新
            rotation_transform = transforms.Affine2D().rotate_deg_around(0, 0, np.rad2deg(heading) + 90)
            translation_transform = transforms.Affine2D().translate(x, y)
            transform = rotation_transform + translation_transform + self.ax.transData
            self.auv_patches[i].set_transform(transform)

            if len(self.positions[i]) > 1:
                positions = np.array(self.positions[i])
                self.ax.plot(positions[:, 0], positions[:, 1], color=colors[i])

        # x, y = self.state[0], self.state[1]
        # heading = 0
        # rotation_transform = transforms.Affine2D().rotate_deg_around(0, 0, np.rad2deg(heading) + 90)
        # translation_transform = transforms.Affine2D().translate(x, y)
        # transform = rotation_transform + translation_transform + self.ax.transData
        # self.auv_patch.set_transform(transform)
        # if len(self.positions) > 1:
        #     positions = np.array(self.positions)
        #     self.ax.plot(positions[:, 0], positions[:, 1], color='blue')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None

gym.envs.registration.register(
    id="AuvEnv-v0",
    # entry_point='auv_env:AuvEnv',
    entry_point=AuvEnv,
)
