import gymnasium as gym
import genesis as gs
import numpy as np

from gym_genesis.tasks.cube import CubeTask
class GenesisEnv(gym.Env):

    def __init__(
            self,
            task,
    ):
        super().__init__()
        self.task = task
        self._env = self._make_env_task(self.task)
        # add action space (TODO: check if compatible)
        self.action_space = self._env.action_space

        # === Set up Genesis scene (task-specific env will populate it) ===
        gs.init(backend=gs.gpu, precision="32")
        self.scene = None  # Will be created in the child class

    def render(self):
        raise NotImplementedError("not implemented yet")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        observation = self._env.reset()

        info = {"is_success": False}
        return observation, info
    
    def step(self, action):
        _, reward, _, raw_obs = self._env.step(action)

        terminated = is_success = reward == 4

        info = {"is_success": is_success}

        observation = self._format_raw_obs(raw_obs)

        truncated = False
        return observation, reward, terminated, truncated, info
    
    def close(self):
        pass
    
    def _get_obs(self):
        return self._env.get_obs()
    
    def _make_env_task(self, task_name):
        if task_name == "cube":
            task = CubeTask()
        else:
            raise NotImplementedError(task_name)
        return task
        
