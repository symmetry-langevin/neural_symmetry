import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from functools import partial
#from jax.lib import xla_bridge
#print(xla_bridge.get_backend().platform)

def make_grid(res, xrange=2):
    grid = jnp.mgrid[:res, :res].transpose(1, 2, 0)
    grid = (grid / (res - 1) * 2 - 1) * xrange
    return grid

def safe_norm(v, axis=None, eps=1e-6):
    l = jnp.linalg.norm(v, axis=axis)
    return jnp.where(l < eps, 0, l)

def safeguard(val, eps=1e-6):
        return jnp.where(jnp.isnan(val), 0, val)
    
def rot_mat(theta):
    return jnp.array([
        [jnp.cos(theta), jnp.sin(theta)],
        [-jnp.sin(theta), jnp.cos(theta)]
    ])

def find_tangent_points(X, R=1, eps=1e-7):
    # Calculate length for X
    
    dist_X = jnp.maximum(safe_norm(X), eps)
    # Calculate direction vector for OX
    dx = safeguard(X / dist_X)

    # Calculate angle theta for the tangent points
    theta = jnp.arccos(jnp.clip(safeguard(R / dist_X), min=eps-1, max=1-eps))

    # Rotate vector by +theta and -theta to find tangent points
    rotation_matrix_cw = rot_mat(theta)
    rotation_matrix_ccw = rot_mat(-theta)
    T1 = jnp.matmul(rotation_matrix_cw, dx[:, None])[:, 0] * R
    T2 = jnp.matmul(rotation_matrix_ccw, dx[:, None])[:, 0] * R
    L = safe_norm(X - T1)
    L = jnp.where(L < eps, 0, L)
    return T1, T2, L

def calculate_arc_length(v1, v2, R=1, eps=1e-7):
    # Calculate angle between T1 and T2 using dot product
    angle = jnp.arccos(
        jnp.clip(
            jnp.dot(v1, v2) / R**2,
            min=-1 + eps, max=1 - eps
        )
    )
    angle = jnp.minimum(jnp.pi * 2 - angle, angle)
    return R * angle

def calculate_path_on_ball(X, Y, R=1):
    # Get the tangent points from X and Y to the circle
    T1_X, T2_X, tx_len = find_tangent_points(X, R=R)
    T1_Y, T2_Y, ty_len = find_tangent_points(Y, R=R)
    # Compute path lengths for different combinations of tangents
    arc_length_1 = calculate_arc_length(T1_X, T1_Y, R=R)
    arc_length_2 = calculate_arc_length(T1_X, T2_Y, R=R)
    arc_length_3 = calculate_arc_length(T2_X, T1_Y, R=R)
    arc_length_4 = calculate_arc_length(T2_X, T2_Y, R=R)
    arc_length = jnp.min(
        jnp.array([arc_length_1, arc_length_2, arc_length_3, arc_length_4]))
    # print(arc_length)
    return arc_length + tx_len + ty_len

def intersect_wormshole(x, y, R=1, eps=1e-8):
    d = safeguard((y - x) / (safe_norm(y - x)))
    n = jnp.array([-d[1], d[0]])
    tp = - jnp.dot(x, d)
    ty = safe_norm(x - y)
    t = jnp.maximum(jnp.minimum(ty, tp), 0)
    r = safe_norm(x + t * d)
    return r < R

def dist_outside_wormhole(x, y, R=1):
    return jnp.where(
        jax.lax.stop_gradient(intersect_wormshole(x, y, R=R)),
        calculate_path_on_ball(x, y, R=R),
        safe_norm(x - y))
    # return calculate_path_on_ball(x, y, R=R)
    # return jnp.linalg.norm(x - y)
    
def dist_through_wormhole(x, y, R=1):
    def _norm_(v):
        return safeguard(v / safe_norm(v))
    z = _norm_(0.5*(x-y)) * R
    return safe_norm(x-z) + safe_norm(y+z)

def inside_wormhole(x, R=1):
    return safe_norm(x) < R

# @jax.jit
def geodesic_dist(x, y, R=1):
    xvalid = jnp.logical_not(inside_wormhole(x, R=R))
    yvalid = jnp.logical_not(inside_wormhole(y, R=R))
    valid = jnp.logical_and(xvalid, yvalid)
    return jnp.where(
        jax.lax.stop_gradient(valid),
        jnp.min(jnp.array([
        dist_through_wormhole(x, y, R=R),
        dist_outside_wormhole(x, y, R=R),
        ])),
        -1
    )

def weighted_dist(x, ys, R=1, sigma=0.1):
    """
    [x] : (2,)
    [ys]: (N, 2)
    """
    # print(x.shape, ys.shape)
    dists = jax.vmap(
        partial(geodesic_dist, R=R),
        in_axes=(None, 0)
    )(x, ys)
    weights = jnp.exp(-(dists / sigma)**2)
    return weights.mean()


def grad_geodesic(x, y, R=1):
    return jax.grad(lambda p: geodesic_dist(p, y, R=R).sum())(x)

def weighted_grad(x, ys, R=1, sigma=0.1, eps=1e-6):
    """
    [x] : (2,)
    [ys]: (N, 2)
    """
    dists = jax.vmap(
        partial(geodesic_dist, R=R),
        in_axes=(None, 0)
    )(x, ys)
    weights = jax.nn.softmax(-(dists / sigma)**2)

    grads = jax.vmap(
        partial(grad_geodesic, R=R),
        in_axes=(None, 0)
    )(x, ys) # (N, 2)
    grads_norm = safe_norm(grads, axis=-1) ** 2
    grads = safeguard(grads / (grads_norm[:, None])) * dists[:, None]
    w_grads = (grads * weights.reshape(-1, 1)).sum(axis=0)
    return w_grads

def get_prior(key, num_points, inp_dim, r_min, r_max):
    random_values = random.uniform(key, shape=(num_points, inp_dim))
    scaled_values = random_values * (r_max - r_min) + r_min
    return scaled_values


def contact_circle(x, grad_val, R=1, eps=1e-6):
    target = x + grad_val
    inside = safe_norm(target) < R
    d = safeguard(grad_val / safe_norm(grad_val))
    t_M = jnp.dot(-x, d)
    M = x + t_M * d
    lM = safe_norm(M)
    theta = jnp.arccos(jnp.clip(lM / R, min=eps-1, max=1-eps))
    M_norm = safeguard(M / lM) * R # contact point should be on sphere with R
    Y_pos = jnp.matmul(rot_mat(theta), M_norm)
    Y_neg = jnp.matmul(rot_mat(-theta), M_norm)

    Y_pos = jnp.where(lM < eps, d, Y_pos)
    Y_neg = jnp.where(lM < eps, -d, Y_neg)

    T_Ypos = jnp.dot(Y_pos - x, d)
    T_Yneg = jnp.dot(Y_neg - x, d)

    T_y = jnp.minimum(T_Ypos, T_Yneg)
    Y = x + T_y * d
    vY = target - Y
    return inside, Y, vY


def walk(x, g, R=1, eps=1e-6):
    # Found the contact point
    target_ = x + g
    inside, y, vy = contact_circle(x, g, R=R)

    # Decomposing the tensor
    vN = jnp.dot(vy, y) * y / R
    vT = vy - vN

    # Compute the tagent 
    theta = safe_norm(vT) / R # angle
    theta_sign = jnp.sign(jnp.cross(safeguard(vT/safe_norm(vT)), y/R))
    y_rot = jnp.matmul(rot_mat(theta * theta_sign), y)
    y_rfl = - y_rot
    target_rfl = y_rfl + safeguard(y_rfl / safe_norm(y_rfl)) * safe_norm(vN)
    return jnp.where(inside, target_rfl, target_)


def walk_legacy(x, grad_val, R=1, eps=1e-6):
    target = x + grad_val
    inside = jnp.linalg.norm(target) < R
    d = grad_val / jnp.linalg.norm(grad_val)
    t_M = jnp.dot(-x, d)
    M = x + t_M * d
    lM = jnp.linalg.norm(M)
    theta = jnp.arccos(jnp.clip(lM / R, min=eps-1, max=1-eps))
    M_norm = M / lM
    Y_pos = jnp.matmul(rot_mat(theta), M_norm)
    Y_neg = jnp.matmul(rot_mat(-theta), M_norm)

    Y_pos = jnp.where(lM < eps, d, Y_pos)
    Y_neg = jnp.where(lM < eps, -d, Y_neg)

    T_Ypos = jnp.dot(Y_pos - x, d)
    T_Yneg = jnp.dot(Y_neg - x, d)

    T_y = jnp.minimum(T_Ypos, T_Yneg)
    Y = x + T_y * d
    Y_out = -Y
    vY = target - Y
    target_alt = Y_out + vY # vY might need to be transformed
    return jnp.where(inside, target_alt, target)