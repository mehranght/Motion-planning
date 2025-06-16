import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import pybullet as p
import pybullet_data

from pybullet_planning import connect, disconnect, \
    get_movable_joints, get_links, set_joint_positions, \
    wait_if_gui, plan_joint_motion1, plan_joint_motion, get_collision_fn
from simulation.robots import get_robot, get_ik_solution
import yaml

from simulation.environment import get_obstacles
from simulation.planning import plan_path
from simulation.robots import get_robot
data = torch.load('/home/mehran/motion-planning/MotionPlanning_UR5/val_data_place.pt')



# print(data['paths'])


# torch.save(
#         {
#             'paths': data['paths'][0:13000]
#         },
#         '/home/mehran/motion-planning/MotionPlanning_UR5/train_place_bp.pt')

# torch.save(
#         {
#             'paths': data['paths'][13000:15000]
#         },
#         '/home/mehran/motion-planning/MotionPlanning_UR5/val_place_bp.pt')




path = data['paths'][0].detach().numpy()

with open('/home/mehran/motion-planning/MotionPlanning_UR5/configs/basic.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

connect(use_gui=True)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
group = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6']
robot = p.loadURDF(
        '/home/mehran/motion-planning/MotionPlanning_UR5/simulation/ur5.urdf',
        [0, 0, 0],
        flags=p.URDF_USE_SELF_COLLISION)
obstacles = get_obstacles(**cfg['bin'])
body_link = get_links(robot)[-1]
joints1 = get_movable_joints(robot)
print(joints1)
joints = dict(zip(group,joints1))
# collision_fn = get_collision_fn(
#         robot,
#         joints1,
#         obstacles)
# goal_xyz = np.array([0.28341294, -0.40423531, -0.14573738])
# goal_orientation = np.array([0.66328088, 0.4121298,  0.33213266, 0.52905141])

# goal_configuration = get_ik_solution(
#                 robot,
#                 joints,
#                 goal_xyz,
#                 goal_orientation,
#                 collision_fn)

# set_joint_positions(robot, joints1, goal_configuration)

for i, conf in enumerate(path):
        set_joint_positions(robot, joints1, conf.reshape(6))
        wait_if_gui('Step: {}/{}'.format(i, len(path)))
wait_if_gui('Finish?')
disconnect()