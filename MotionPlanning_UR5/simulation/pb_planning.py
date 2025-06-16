from pybullet_planning.interfaces.planner_interface import (
    MAX_DISTANCE,
    get_sample_fn,
    get_distance_fn,
    get_extend_fn,
    get_collision_fn,
    get_joint_positions,
    check_initial_end,
    birrt)


class Wrapper:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)


def plan_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, diagnosis=False, **kwargs):
    """call birrt to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
    """
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)
    collision_fn = Wrapper(collision_fn)

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn, diagnosis=diagnosis):
        return None
    out = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
    return out, collision_fn.count
