import gymnasium as gym
class GenesisEnv(gym.Env):

    def __init__(
            self
    ):
        super().__init__()

    def render(self):
        raise NotImplementedError("not implemented yet")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError("not implemented yet.") 
    
    def step(self, action):
        raise NotImplementedError("not implemented yet")
    
    def close(self):
        raise NotImplementedError("not implemented yet")