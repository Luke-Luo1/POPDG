import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, \
                                 matrix_to_rotation_6d, rotation_6d_to_matrix

def ax_to_6v(q):
    """ Converts an axis-angle tensor to a 6D rotation representation. """
    assert q.shape[-1] == 3, "Input tensor must be of shape (*, 3)"
    return matrix_to_rotation_6d(axis_angle_to_matrix(q))

def ax_from_6v(q):
    """ Converts a 6D rotation tensor back to an axis-angle representation. """
    assert q.shape[-1] == 6, "Input tensor must be of shape (*, 6)"
    return matrix_to_axis_angle(rotation_6d_to_matrix(q))

def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 9)
    :param y: quaternion tensor (N, S, J, 9)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res
