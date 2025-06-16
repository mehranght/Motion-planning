import atexit
import contextlib
import mock


from scipy.spatial.transform import Rotation
import numpy as np
import pybullet
from pybullet_planning import \
    disconnect, connect, get_movable_joints, get_collision_fn, set_joint_positions
from pybullet_utils.bullet_client import BulletClient
from simulation.bin import create_bin
from simulation.robots import get_robot, CUSTOM_LIMITS


def get_obstacles(p, dims, location):
    bin_pose = get_bin_pose_4x4(np.array(location))
    return create_bin(p, np.array(dims), bin_pose, 0.001)


def get_bin_rotation():
    return Rotation.from_euler('zyx', [-90, 0, 0], degrees=True)


def get_bin_pose_4x4(bin_location):
    bin_transform = np.eye(4)
    bin_transform[:3, :3] = get_bin_rotation().as_matrix()
    bin_transform[:3, 3] = bin_location
    return bin_transform


@contextlib.contextmanager
def patch_pybullet_planning_client(client_id):
    with (
        mock.patch(
            'pybullet_planning.interfaces.control.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.debug_utils.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.geometry.bounding_box.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.geometry.camera.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.env_manager.pose_transformation.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.env_manager.savers.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.env_manager.shape_creation.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.env_manager.simulation.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.env_manager.user_io.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.kinematics.ik_utils.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.robots.body.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.robots.collision.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.robots.dynamics.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.robots.joint.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.interfaces.robots.link.CLIENT',
            client_id),
        mock.patch(
            'pybullet_planning.CLIENT',
            client_id),
    ):
        yield


class Environment:
    def __init__(self, cfg, visualize):
        self.cfg = cfg
        self.visualize = visualize

        if visualize:
            connect(use_gui=visualize)
            self.p = pybullet
            # I don't know if this matters
            atexit.register(disconnect)
        else:
            self.p = BulletClient(connection_mode=pybullet.DIRECT)

        self.p.setAdditionalSearchPath('simulation/',
                                       physicsClientId=self.physics_client_id)
        self.p.setAdditionalSearchPath('simulation/table/',
                                       physicsClientId=self.physics_client_id)

        self.obstacles = get_obstacles(self.p, **cfg['bin'])

        self.robot = get_robot(self.p)

        with patch_pybullet_planning_client(self.physics_client_id):
            self.joints = get_movable_joints(self.robot)
            self._collision_fn = get_collision_fn(
                self.robot,
                self.joints,
                self.obstacles)

        self.custom_limits = {j: l for j, l in zip(self.joints, CUSTOM_LIMITS)}

        self.collision_check_counter = 0

    @property
    def physics_client_id(self):
        return getattr(self.p, '_client', 0)

    def collision_fn(self, *args):
        self.collision_check_counter += 1
        with patch_pybullet_planning_client(self.physics_client_id):
            return self._collision_fn(*args)

    def set_joint_values(self, values):
        with patch_pybullet_planning_client(self.physics_client_id):
            set_joint_positions(self.robot, self.joints, values)

    def get_link_state(self, link_id):
        return self.p.getLinkState(
            self.robot, link_id, physicsClientId=self.physics_client_id)
