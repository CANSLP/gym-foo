import time
import random

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math

import gym
from gym.envs.classic_control import rendering

from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from gym.envs.registration import register


class JumperEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        #game variables
        self.gametick = 0

        self.win = 0
        self.reward = 0.0



        self.baseHeight = 0.1

        self.goalX = random.randint(2,8)/10
        self.goalY = 0.9

        self.playerX = 0.5
        self.playerY = 0.2
        self.playerXv = 0
        self.playerYv = 0
        self.pxs = 0.002
        self.pJump = 0.05

        self.gravity = -0.003

        self.plat1X = random.randint(3,17)/20
        self.plat2X = random.randint(3, 17) / 20
        self.plat3X = random.randint(3, 17) / 20
        self.plat4X = random.randint(3, 17) / 20

        self.closest = math.sqrt(math.pow((self.playerX - self.goalX), 2) + math.pow((self.playerY - self.goalY), 2))


        high = np.array([self.win,self.closest,0,0])


        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

        #game mechanics
        self.gametick+=1

        self.playerXv*=0.5

        if abs(self.playerX-0.5) > 0.5:
            self.pxs = self.pxs*-1
            self.playerX+=self.pxs
            self.playerXv*=-1
            if action == 1:
                self.playerYv = self.pJump * 0.75



        self.playerYv+=self.gravity
        if action == 0:
            self.playerXv+=self.pxs*3


        self.playerX+=self.playerXv
        self.playerY+=self.playerYv

        if self.playerY<self.baseHeight:
            self.playerY = self.baseHeight
            if action == 1:
                self.playerYv = self.pJump
            else:
                self.playerYv = 0

        if self.playerY>0.25+(self.playerYv-0.01) and self.playerY < 0.25 and abs(self.playerX-self.plat1X) < 0.15:
            self.playerY = 0.25
            if action == 1:
                self.playerYv = self.pJump
            else:
                self.playerYv = 0

        if self.playerY>0.4+(self.playerYv-0.01) and self.playerY < 0.4 and abs(self.playerX-self.plat2X) < 0.15:
            self.playerY = 0.4
            if action == 1:
                self.playerYv = self.pJump
            else:
                self.playerYv = 0

        if self.playerY>0.55+(self.playerYv-0.01) and self.playerY < 0.55 and abs(self.playerX-self.plat3X) < 0.15:
            self.playerY = 0.55
            if action == 1:
                self.playerYv = self.pJump
            else:
                self.playerYv = 0

        if self.playerY>0.7+(self.playerYv-0.01) and self.playerY < 0.7 and abs(self.playerX-self.plat4X) < 0.15:
            self.playerY = 0.7
            if action == 1:
                self.playerYv = self.pJump
            else:
                self.playerYv = 0

        if math.sqrt(math.pow((self.playerX-self.goalX),2)+math.pow((self.playerY-self.goalY),2)) < 0.05:
            self.playerX = self.goalX
            self.playerY = self.goalY
            self.playerXv = 0
            self.playerYv = 0
            self.gravity = 0
            self.pxs = 0
            self.closest = 0
            self.win+=1

        if math.sqrt(math.pow((self.playerX-self.goalX),2)+math.pow((self.playerY-self.goalY),2)) < self.closest:
            self.closest = self.closest = math.sqrt(math.pow((self.playerX-self.goalX),2)+math.pow((self.playerY-self.goalY),2))




        done = False

        self.reward = (self.win/10)+(1/(math.pow(self.closest,2)+0.01))
        reward = self.reward

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

        print("reset")

        # set initial game variable values
        self.playerX = 0.5
        self.playerY = 0.2
        self.playerXv = 0
        self.playerYv = 0
        self.pxs = 0.002
        self.gametick = 0
        self.reward = 0.0

    def render(self, mode='human'):
        #print("render")

        screen_width = 300
        screen_height = 300



        #rendering game variables

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if True:
            # render game objects
            #print("draw")

            # self.Object = rendering.Function(?,?,...)
            # self.viewer.add_geom(self.Object)

            # self.Object.set_color(r,g,b)

            # rendering.
            # FilledPolygon([(,),(,),...])
            # Transform()
            # make_circle()
            # Line((,),(,))

            # example orange background
            self.back = rendering.FilledPolygon([(0, 0), (screen_width, 0), (screen_width, screen_height), (0, screen_height)])
            self.back.set_color(1, 0.5, 0)
            self.viewer.add_geom(self.back)

            self.base = rendering.FilledPolygon([(0, screen_height * self.baseHeight), (screen_width, screen_height * self.baseHeight),(screen_width, 0), (0, 0)])
            self.base.set_color(0.1, 0.05, 0)
            self.viewer.add_geom(self.base)

            self.goal = rendering.make_circle((screen_width + screen_height) * 0.02)
            self.goalTran = rendering.Transform(translation=(self.goalX * screen_width, self.goalY * screen_height))
            self.goal.add_attr(self.goalTran)
            self.goal.set_color(1, 1, 0.25)
            self.viewer.add_geom(self.goal)

            self.player = rendering.make_circle((screen_width + screen_height) * 0.01)
            self.playerTran = rendering.Transform(translation=(self.playerX * screen_width, self.playerY * screen_height))
            self.player.add_attr(self.playerTran)
            self.player.set_color(1, 0.25, 0)
            self.viewer.add_geom(self.player)

            self.plat1 = rendering.Line(((self.plat1X-0.15)* screen_width,0.25* screen_height),((self.plat1X+0.15)* screen_width,0.25* screen_height))
            self.plat1.set_color(0.1,0.05,0)
            self.viewer.add_geom(self.plat1)

            self.plat2 = rendering.Line(((self.plat2X - 0.15) * screen_width, 0.4 * screen_height),((self.plat2X + 0.15) * screen_width, 0.4 * screen_height))
            self.plat2.set_color(0.1, 0.05, 0)
            self.viewer.add_geom(self.plat2)

            self.plat3 = rendering.Line(((self.plat3X - 0.15) * screen_width, 0.55 * screen_height),((self.plat3X + 0.15) * screen_width, 0.55 * screen_height))
            self.plat3.set_color(0.1, 0.05, 0)
            self.viewer.add_geom(self.plat3)

            self.plat4 = rendering.Line(((self.plat4X - 0.15) * screen_width, 0.7 * screen_height),((self.plat4X + 0.15) * screen_width, 0.7 * screen_height))
            self.plat4.set_color(0.1, 0.05, 0)
            self.viewer.add_geom(self.plat4)


        if self.state is None: return None


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        print(self.reward)
        if self.viewer: self.viewer.close()


jumper = JumperEnv();
jumper.reset()
for tick in range(200):
    ac = 0
    if random.randint(0,5) < 2:
        ac = random.randint(0,1)
        jumper.step(ac)
        jumper.render()
    time.sleep(1/50)

jumper.close()
