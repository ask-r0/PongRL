from src.pong_env_manager import PongEnvManager
import cv2

env_manager = PongEnvManager(4, enable_render=False)
env_manager.reset()
for i in range(1000):
    env_manager.step(env_manager.get_random_action())
    if i == 2:
        print("HELLO")
        s = env_manager.get_processed_state()
        for j in range(s.shape[0]):
            cv2.imwrite(f"{j}.png", s[j])
