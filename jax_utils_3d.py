from jax_utils import *
from jax.scipy.spatial.transform import Rotation

def find_plane_rotation(x,y):
    """
    finds a plane that contains x, y, origin
    returns a rotation matrix to map x and y to this plane
    
    inputs:
    x:  (3,)
    y:  (3,)
    outputs: 
    R:  (3,3)
    """   

    n1 = jnp.cross(x, y)
    n2 = jnp.array([0,0,1]) #z axis
    r1_axis = jnp.cross(n1, n2)
    eps = 1e-8
    r1_axis = safeguard(r1_axis / safe_norm(r1_axis))
    cos = jnp.clip(safeguard(jnp.dot(n1, n2) / (safe_norm(n1) * safe_norm(n2))), -1, 1)
    # rotation aligning the normal directions
    rotmat = Rotation.from_rotvec(r1_axis * jnp.acos(cos))
    #q_rot = jnp.matmul(rotmat, x)
    #t_rot = jnp.matmul(rotmat, y)
    #rotmat_inv = jnp.transpose(rotmat)
    return jnp.where(jnp.abs(safe_norm(jnp.cross(n1, n2)))< eps, Rotation.identity().as_matrix(), rotmat.as_matrix())

def from_3d(x,rotmat):
    """
    with the rotation matrix found, rotates x such that z axis is removed
    inputs:
    x:  (3,)
    R:  (3,3)
    outputs: 
    R(x): (2,)
    """      
    rot = Rotation.from_matrix(rotmat)
    return rot.apply(x)[:2] #always take z dim out

def lift_to_3d(x, rotmat):
    """
    inputs:
    x:  (2,)
    R:  (3,3)
    outputs: 
    R(x): (3,)
    """      
    rot = Rotation.from_matrix(rotmat)
    rotmat_inv = rot.inv()
    stacked = jnp.append(x,0)
    return rotmat_inv.apply(stacked)

def weighted_dist_3d(x, ys, R=1, sigma=0.1):
    """
    [x] : (3,)
    [ys]: (N, 3)
    """
    dists = jax.vmap(
        partial(geodesic_dist_3d, R=R),
        in_axes=(None, 0)
    )(x, ys)
    weights = jnp.exp(-(dists / sigma)**2)
    return weights.mean()

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

def geodesic_dist_3d(x, y, R=1):
    """
    inputs:
    x:    (3,)
    y:    (3,)
    outputs: 
    dist: (1,)
    """

    rotmat = find_plane_rotation(x,y)
    x_2d = from_3d(x, rotmat)
    y_2d = from_3d(y, rotmat)
    dist_2d = geodesic_dist(x_2d, y_2d, R=1)

    return dist_2d
    
def grad_geodesic_3d(x, y, R=1):
    return jax.grad(lambda p: geodesic_dist_3d(p, y, R=R).sum())(x)

def weighted_grad_3d(x, ys, R=1, sigma=0.1, eps=1e-6):
    """
    [x] : (3,)
    [ys]: (N, 3)
    """
    dists = jax.vmap(
        partial(geodesic_dist_3d, R=R),
        in_axes=(None, 0)
    )(x, ys)
    
    weights = jax.nn.softmax(-(dists / sigma)**2)
    grads = jax.vmap(
        partial(grad_geodesic_3d, R=R),
        in_axes=(None, 0)
    )(x, ys)

    grads_norm = jnp.linalg.norm(grads, axis=-1) ** 2
    grads = safeguard(grads / (grads_norm[:, None] + eps)) * dists[:, None]
    w_grads = (grads * weights.reshape(-1, 1)).sum(axis=0)
    return w_grads

def walk_3d(x, g, R=1, eps=1e-6):
    """
    input: 
    x: (3,)
    g: (3,)
    
    output:
    target: (3,)
    """

    target_ = x + g
    R = find_plane_rotation(x, target_)
    x_rot = from_3d(x, R)
    g_rot = from_3d(g, R)
    
    walked = walk(x_rot, g_rot)
    walked_3d = lift_to_3d(walked, R)
    return walked_3d

def batch_dist_custom(x, t_pcl, sigma=1., batch_size=100):
    
    bs = x.shape[:-1]
    xf = x
    out = []
    dist_fn = jax.vmap(
        partial(weighted_dist_3d, R=1, sigma=sigma),
        in_axes=(0, None)
    )
    for i in range(0, xf.shape[0], batch_size):
        j = min(i + batch_size, xf.shape[0])
        xb = xf[i:j]
        dist_val = dist_fn(xb, t_pcl)
        out.append(dist_val)
    return jnp.stack(out).reshape(*bs)

# def rotmat_from_vec(rotation_vector, eps = 1e-8):
#     theta = safe_norm(rotation_vector)

#     k = rotation_vector / (theta+eps)
#     K = jnp.array([[0, -k[2], k[1]],
#                   [k[2], 0, -k[0]],
#                   [-k[1], k[0], 0]])

#     # Rodrigues' rotation formula
#     R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
    
#     return jnp.where(jnp.abs(theta) < eps, jnp.identity(3), R)

# def rotate_align(query, target):
#     """
#     queyr: (3,)
#     target: (N,3)
#     """
#     n1 = jnp.cross(query, target)
#     n2 = jnp.array([0,0,1]) #z axis

#     r1_axis = jnp.cross(n1, n2)
#     if (jnp.abs(safe_norm(r1_axis)) < 1e-8) or (jnp.abs(safe_norm(n1)) < 1e-8) or (jnp.abs(safe_norm(n2)) < 1e-8):
#         rotmat = Rotation.identity()
#     else:
#         r1_axis = r1_axis / safe_norm(r1_axis)
#         cos = jnp.dot(n1, n2) / (safe_norm(n1) * safe_norm(n2))
#         # rotation aligning the normal directions
#         cos = jnp.clip(cos, -1, 1)
#         rotmat = Rotation.from_rotvec(r1_axis * jnp.acos(cos))
#     q_rot = rotmat.apply(query)
#     t_rot = rotmat.apply(target)
#     rotmat_inv = rotmat.inv()
#     return q_rot, t_rot, rotmat_inv    