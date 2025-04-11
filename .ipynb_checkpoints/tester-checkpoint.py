from gym_genesis.env import GenesisEnv 
import genesis as gs

# first initialize the backend
gs.init(backend=gs.gpu, precision="32")

env = GenesisEnv(task="cube", enable_pixels=True)
obs, _ = env.reset()
breakpoint()
print(obs["pixels"].shape)  # Should be (960, 1280, 3)
