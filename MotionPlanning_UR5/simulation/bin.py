import numpy as np
from scipy.spatial.transform import Rotation


def create_bin(p, bin_dims, bin_pose, thickness):
    """ Create bin multibodies with a given dimension, pose, and wall thickness

    :param bin_dims: size in x, size in y, size in z measured in m
    :param bin_pose: 4x4 pose of the bin in world space
    :param thickness: thickness of each wall and floor in m
    :return: list of multibody ids
    """
    physics_client_id = getattr(p, '_client', 0)

    half_bin_dims = bin_dims / 2

    # Call the "unoriented" space, a space where the box is axis aligned with
    # the middle of the bottom at the origin.
    # There are four boxes that make up the bin, four walls and a base.
    # Below are the coordinates of the middles of those four boxes in the
    # unoriented space.
    unoriented_middles = np.array([
        (-half_bin_dims[0], 0, half_bin_dims[2]),
        (half_bin_dims[0],  0, half_bin_dims[2]),
        (0, -half_bin_dims[1], half_bin_dims[2]),
        (0, half_bin_dims[1], half_bin_dims[2]),
        (0, 0, 0)  # base
    ])

    # Apply the bin rotation to go from "unoriented" space to "oriented" space.
    # The bottom middle of the bin is still at (0, 0, 0), but the middle of
    # each wall has moved.
    oriented_middles = (bin_pose[:3, :3] @ unoriented_middles.T).T

    # The dimensions of each box composing the walls and the floor
    dims = np.array([
        (thickness, bin_dims[1], bin_dims[2]),
        (thickness, bin_dims[1], bin_dims[2]),
        (bin_dims[0], thickness, bin_dims[2]),
        (bin_dims[0], thickness, bin_dims[2]),
        (bin_dims[0], bin_dims[1], thickness)
    ])

    bin_position = bin_pose[:3, 3]
    bin_orientation = Rotation.from_matrix(bin_pose[:3, :3]).as_quat()

    out = []
    for middle, dim in zip(oriented_middles, dims):
        # Create collision and visual shapes
        collision_id = p.createCollisionShape(
            halfExtents=dim / 2,
            shapeType=p.GEOM_BOX,
            physicsClientId=physics_client_id)

        visual_shape_id = p.createVisualShape(
            halfExtents=dim / 2,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            shapeType=p.GEOM_BOX,
            physicsClientId=physics_client_id)

        # Create a multibody, here basePosition is our chance to translate the
        # GEOM_BOX we created above from (0, 0, 0) to where it's middle should
        # be, and the baseOrientation is our chance to rotate that box about
        # its middle.
        muiltibody_id = p.createMultiBody(
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=middle + bin_position,
            baseOrientation=bin_orientation,
            physicsClientId=physics_client_id)

        out.append(muiltibody_id)

    return out
