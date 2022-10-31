import collections
import gym
import numpy as np

from src.frame_processer import ImagePreProcessor


class PongEnvManager:
    def __init__(self, screen_buffer_size, enable_render):
        if enable_render:
            self.env = gym.make("ALE/Pong-v5", render_mode="human")
        else:
            self.env = gym.make("ALE/Pong-v5")

        self.img_processor = ImagePreProcessor(84, 25, 5, perform_normalization=True)

        self.screen_buffer_size = screen_buffer_size
        self.screen_buffer = collections.deque(maxlen=screen_buffer_size)
        self.__reset_screen_buffer()

        self.is_done = False
        self.time = 0

    def reset(self):
        self.env.reset()
        #  self.__reset_screen_buffer()
        self.time = 0
        self.is_done = False

    def step(self, action):
        observation, reward, done, info = self.env.step(action)  # TODO: verify the return types and meanings
        self.__update_buffer(observation)
        self.is_done = done
        self.time += 1

        return reward

    def get_processed_state(self):
        """ Returns a processed state of the environment.
        An environment state is represented by 4 consecutive frames.
        4 consecutive frames is necessary to capture direction of motion, speed, etc.

        Each frame is processed by (1) grayscale frame (2) crop frame.
        NB! preprocessing not done in this method, but rather when added to screen_buffer
        """
        frames = []
        for i in range(self.screen_buffer_size):
            frames.append(self.screen_buffer[i])
        return np.stack(frames)

    def get_random_action(self):
        return self.env.action_space.sample()

    def __reset_screen_buffer(self):
        self.screen_buffer.clear()

        for i in range(self.screen_buffer_size):
            self.screen_buffer.append(self.img_processor.get_black_image())

    def __update_buffer(self, screen):
        screen = self.img_processor.get_processed_img(screen)
        self.screen_buffer.append(screen)
