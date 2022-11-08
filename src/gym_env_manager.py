import gym
import collections
import numpy as np
import cv2


class GymEnvManager:
    def __init__(self, skip_frames, gym_environment_name, render):
        if render:
            self.env = gym.make(gym_environment_name, render_mode="human")
        else:
            self.env = gym.make(gym_environment_name)

        self.obs_buffer = collections.deque(maxlen=2)
        self.skip_frames = skip_frames

        self.do_normalize = True

    def reset(self):
        self.obs_buffer.clear()
        obs1 = self.env.reset()
        obs2, _, _, _ = self.env.step(1)  # Performs fire needed to start game

        processed_obs1 = self.process_observation(obs1, self.do_normalize)
        processed_obs2 = self.process_observation(obs2, self.do_normalize)
        self.obs_buffer.append(processed_obs1)
        self.obs_buffer.append(processed_obs2)

        max_frames = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frames

    def step(self, action):
        reward_sum = 0
        done = None
        for i in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            processed_obs = self.process_observation(obs, self.do_normalize)
            self.obs_buffer.append(processed_obs)
            reward_sum += reward
            if done:
                break
        max_frames = np.max(np.stack(self.obs_buffer), axis=0)  # Max of last two frames
        return max_frames, reward_sum, done, info

    def get_random_action(self):
        return self.env.action_space.sample()

    @staticmethod
    def process_observation(frame, do_normalize):
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale
        frame = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
        frame = frame[33:(frame.shape[0] - 15), :]  # crop
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # resize to 84x84
        if do_normalize:
            frame = np.array(frame).astype(np.float32) / 255.0
        return frame
