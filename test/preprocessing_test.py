from src.utils.gym_env_manager import GymEnvManager
import cv2


def preprocessing_test(filename):
    env = GymEnvManager(4, "BreakoutNoFrameskip-v4", False, False, 33, 15)  # Env does not matter, just has to be valid
    processed = env.process_observation(cv2.imread(filename))
    start_filename = filename.split(".")[0]
    cv2.imwrite(start_filename+"_processed.png", processed)

