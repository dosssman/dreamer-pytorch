import gym
import collections
import numpy as np
import pybullet_envs

import cv2; cv2.ocl.setUseOpenCL(False)

from dreamer.envs.env import EnvInfo
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.base import Env, EnvStep

# Wrapers for pixel-based tasks
# Wrapper around MJ / Pybullet envs mainly
class StandardPixelBasedWrapper(gym.Wrapper):
    def __init__(self, env, size=(64,64)):
        gym.Wrapper.__init__(self, env)
        self._size = size
        # Inherit some fields from the wrapped env
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = gym.spaces.Box(0,255,(3,) + self._size, dtype=np.uint8)
        self.action_space = self.env.action_space

    def seed(self, seed):
        self.env.seed(seed)
    
    def reset(self):
        self.env.reset()
        obs_img = self.env.render("rgb_array")
        obs_img = np.array(cv2.resize(obs_img, self._size, interpolation=cv2.INTER_AREA)).transpose(2,0,1)
        
        return obs_img
    
    def step(self, action):
        _, rew, done, info = self.env.step(action)
        obs_img = self.env.render("rgb_array")
        obs_img = np.array(cv2.resize(obs_img, self._size, interpolation=cv2.INTER_AREA)).transpose(2,0,1)
        
        return obs_img, rew, done, info
    
    def render(self,mode,*args,**kwargs):
        return self.env.render(mode=mode)
        return self.env.render(*args,*kwargs)

# TODO: consider scaling in the environment directly, with a Wrapper ?
class ScaleWrapper(gym.Wrapper):
    def __init__(self, env, old_min=0., old_max=255., new_min=-1., new_max=1.):
        gym.Wrapper.__init__(self, env)
        self.old_min, self.old_max, self.new_min, self.new_max = \
            old_min, old_max, new_min, new_max
        self.observation_space = gym.spaces.Box(
            low=new_min, high=new_max,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )
        self.action_space = self.env.action_space
        self._max_episode_steps = self.env._max_episode_steps
    
    def _scale_img(self, a):
        return ((a - self.old_min) * (self.new_max - self.new_min)) / (self.old_max - self.old_min) + self.new_min
    
    def reset(self):
        return self._scale_img(self.env.reset())
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._scale_img(obs)
        return obs, reward, done, info
    
# From: https://github.com/denisyarats/pytorch_sac_ae/blob/7fa560e21c026c04bb8dcd72959ecf4e3424476c/utils.py#L143
# Customized a little bit
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        low_arr = np.concatenate([env.observation_space.low for _ in range(k)],0)
        high_arr = np.concatenate([env.observation_space.high for _ in range(k)],0)
        self.observation_space = gym.spaces.Box(
            low=low_arr,
            high=high_arr,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

class PixelBasedBulletEnv(Env):

    def __init__(self, name, size=(64, 64), camera=None):
        self._env = gym.make(name)
        self._size = size

    @property
    def observation_space(self):
        return IntBox(low=0, high=255, shape=(3,) + self._size,
                      dtype="uint8")

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        _, reward, done, info = self._env.step(action)
        obs = self._env.render("rgb_array")
        obs = np.array(cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)).transpose(2,0,1)

        info = EnvInfo(np.array(0.99, np.float32), None, done)
        return EnvStep(obs, reward, done, info)

    def reset(self):
        self._env.reset()
        obs = self._env.render("rgb_array")
        obs = np.array(cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)).transpose(2,0,1)
        
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(*args,*kwargs)
        return self._env.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

    @property
    def horizon(self):
        raise NotImplementedError