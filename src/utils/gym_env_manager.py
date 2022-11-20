import gym
import collections
import numpy as np
import cv2


class GymEnvManager:
    """ Wrapper for a gym environment. Should be used instead of using environment directly
        Attributes:
            skip_frames: Amount of frames to skip per action
            gym_environment_name: Name of the gym environment to be wrapped
            render: True if environment should be visible by humans, else false. NB! Should be false during training
            do_normalize: Should frame be normalized during preprocessing? Should always be True, except when debugging
            crop_from_top: Pixels to crop from top of frame during preprocessing
            crop_from_bottom: Pixels to crop from bottom of frame during preprocessing
            preprocessing_wh_out: Output width and height after preprocessing
    """
    def __init__(self, skip_frames, gym_environment_name, render, do_normalize, crop_from_top, crop_from_bottom,
                 preprocessing_wh_out):
        if render:
            self.env = gym.make(gym_environment_name, render_mode="human")
        else:
            self.env = gym.make(gym_environment_name)

        self.obs_buffer = collections.deque(maxlen=2)  # Used to prevent flickering, see method steps for more docs
        self.skip_frames = skip_frames

        self.do_normalize = do_normalize
        self.crop_from_top = crop_from_top
        self.crop_from_bottom = crop_from_bottom
        self.preprocessing_wh_out = preprocessing_wh_out

    def reset(self):
        """ Resets the environment.
        Observation buffer is cleared, environment reset and
        the action fire is taken (needed to start some gym environments).

        Returns:
            Initial observation. See the documentation for the step method for full explanation of the max_frames line.
        """

        self.obs_buffer.clear()
        obs1 = self.env.reset()
        obs2, _, _, _ = self.env.step(1)  # Performs fire needed to start game

        # Process & store initial observation in observation buffer
        processed_obs1 = self.process_observation(obs1)
        processed_obs2 = self.process_observation(obs2)
        self.obs_buffer.append(processed_obs1)
        self.obs_buffer.append(processed_obs2)

        max_frames = np.max(np.stack(self.obs_buffer), axis=0)
        return max_frames

    def step(self, action):
        """ Performs an action for a number of steps in the environment, dependent on the skip_frames variable.

        According to the paper "Human-level control through deep reinforcement learning" published in Nature,
        flickering is present in some atari games. To remove this the article suggests taking the maximum value for
        each pixel colour value over the two previous frames. This method implements just that, i.e. the observation
        returned from this method is the maximum described in the referenced paper.

        Args:
            action: What action to take for the next skip_frames frames.

         Returns:
             max_frames: Maximum value for each pixel colour value over the two previous processed frames.
               This represents the observation of the environment after performing specified action.
             reward_sum: Sum of rewards from the steps taken.
             done: Boolean indicating if the environment is done (e.g. the game is won/lost).
             info: Info about environment.


        """
        reward_sum = 0
        done = None
        for i in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            processed_obs = self.process_observation(obs)
            self.obs_buffer.append(processed_obs)
            reward_sum += reward
            if done:
                break
        max_frames = np.max(np.stack(self.obs_buffer), axis=0)  # Max of last two frames to prevent flickering
        return max_frames, reward_sum, done, info

    def get_random_action(self):
        """ Returns random action available action to perform in the environment. """
        return self.env.action_space.sample()

    def get_num_actions(self):
        """ Returns number of actions available in environment. """
        return self.env.action_space.n

    def process_observation(self, frame):
        """ Processes a gym environment observation of size 210x160x3 to an observation of size 84x84.
        Done by the following steps:
        1. Grayscale (this removes the rgb channels) (210x160x3 -> 210x160)
        2. Crop image according to crop_from_top and crop_from_bottom attributes
        3. Resize image to 84x84
        4. Normalize image (if enabled)
        """
        frame = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114  # Grayscale
        frame = frame[self.crop_from_top:(frame.shape[0] - self.crop_from_bottom), :]  # Crop
        # Resize
        frame = cv2.resize(frame, (self.preprocessing_wh_out, self.preprocessing_wh_out), interpolation=cv2.INTER_AREA)

        if self.do_normalize:
            frame = np.array(frame).astype(np.float32) / 255.0

        return frame
