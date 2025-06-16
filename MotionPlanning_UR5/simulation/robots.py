import numpy as np

# TODO: What is joint_damping?
JOINT_DAMPING = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Imagine that our robot has a suction tube of length 10 cm sticking out of J6
# that is what we really want to move to positions and orientations inside the
# bin.
# TODO: collision check the tool
TOOL_LENGTH = 0.1

CUSTOM_LIMITS = [
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]


def get_robot(p):
    return p.loadURDF(
        'simulation/ur5.urdf',
        [0, 0, 0],
        flags=p.URDF_USE_SELF_COLLISION,
        physicsClientId=getattr(p, '_client', 0))


def get_ik_solution(
        env,
        goal_xyz,
        goal_orientation,
        retry=1):

    attempt = 0
    while attempt <= retry:
        attempt += 1

        joint_values = env.p.calculateInverseKinematics(
            env.robot,
            9,  # TODO: What is this? end effector index?
            goal_xyz,
            goal_orientation,
            # jointDamping=JOINT_DAMPING,
            solver=0,
            maxNumIterations=1000,
            residualThreshold=.01,
            physicsClientId=env.physics_client_id)

        # Did we achieve the IK goal?
        env.set_joint_values(joint_values)
        link_state = env.p.getLinkState(env.robot, 9,
                                        physicsClientId=env.physics_client_id)
        link_world_position, link_world_orientation = link_state[:2]
        link_world_position = np.array(link_world_position)
        link_world_orientation = np.array(link_world_orientation)
        diff = np.abs(goal_xyz - link_world_position).max()
        if diff > 0.01:
            # print(f'ik failed to achieve position diff {diff}')
            continue
        diff = np.abs(link_world_orientation - goal_orientation).max()
        if diff > 0.01:
            # print(f'ik failed to achieve orientation diff {diff}')
            continue

        # Is it in collision?
        if env.collision_fn(joint_values):
            # print('IK collision')
            continue

        return joint_values

    return None
