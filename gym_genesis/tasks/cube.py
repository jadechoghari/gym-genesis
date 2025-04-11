import genesis as gs
import numpy as np
from gymnasium import spaces

class CubeTask:
    def __init__(self):
        self._build_scene()
        self.random = np.random.RandomState()
        self._random = self.random  # optional, for compatibility with seed()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)


    def _build_scene(self):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(640, 480),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cube = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02))
        )

        self.scene.build()
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.eef = self.franka.get_link("hand")

    def reset(self):
        # Reset robot and cube
        qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
        self.franka.set_qpos(qpos)
        self.scene.step()
        return self.get_obs()

    def step(self, action):
        self.franka.control_dofs_position(action[:7], self.motors_dof)
        self.franka.control_dofs_position(action[7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        raw_obs = self.get_obs()
        return None, reward, None, raw_obs

    def compute_reward(self):
        z = self.cube.get_pos().cpu().numpy()[-1]
        return float(z > 0.1)

    def get_obs(self):
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_rot = self.eef.get_quat().cpu().numpy()
        cube_pos = self.cube.get_pos().cpu().numpy()
        cube_rot = self.cube.get_quat().cpu().numpy()
        gripper = self.franka.get_dofs_position()[7:9].cpu().numpy()

        obs = np.concatenate([
            eef_pos,
            eef_rot,
            cube_pos,
            cube_rot,
            gripper,
            eef_pos - cube_pos,
            np.array([np.linalg.norm(eef_pos - cube_pos)]),
        ])
        return obs