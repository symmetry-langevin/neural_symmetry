import numpy as np
import trimesh
import mediapy as media
from tqdm import tqdm
import plotly.graph_objects as go
from jax_utils_3d import *
from jax_utils import *


#for clustering
def dbscan(D, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        NeighborPts = region_query(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else: 
           C += 1
           grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):    
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = region_query(D, Pn, eps)
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts

        i += 1        
    
def region_query(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        #if geodesic_dist(D[P], D[Pn])<eps:
        if safe_norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors

def compute_centroids(data, labels):
    unique = np.unique(labels)
    unique_labels = unique[unique!=-1]
    centroids = []
    for label in unique_labels:
        mask = labels == label
        points = data[mask]
        centroid = jnp.mean(points, axis=0)
        centroids.append(centroid)
    return np.stack(centroids)

#for plotting and visualization
def create_points_3d(points, alpha=0.3, markersize = 2, label=None):
    Xax = points[:,0]
    Yax = points[:,1] 
    Zax = points[:,2] 

    points3d = go.Scatter3d(
            x=Xax,
            y=Yax,
            z=Zax,
            mode='markers',
            name = label,
            marker=dict(
                size=markersize,
                opacity = alpha
            )
        )
            
    return points3d

def plot_all_3d(to_plot, grid = True):
    fig = go.Figure(data=to_plot)

    # tight layout
    fig.update_scenes(aspectmode='data')
    if not grid:
        fig.update_layout(
        scene = dict(
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
            )
        )
    fig.show()

def plot_plane(normal, point):

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the corresponding z values
    a, b, c = normal
    x0, y0, z0 = point
    z = (a * (x - x0) + b * (y - y0)) / -c + z0
    plane = go.Surface(x=x, y=y, z=z, 
                       colorscale=[[0, '#D8BFD8'], [1, '#D8BFD8']],
                       showscale=False,
                       opacity=0.2)
    return plane

def vis_density(n_points, target_pc, sigma, bs):
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    stacked = np.stack([x_flat, y_flat, z_flat], axis = -1)

    dist_val = batch_dist_custom(stacked, target_pc, sigma=sigma, batch_size=bs)

    normalized_dist_val = (dist_val - 0) / ((dist_val.max() - 0))

    # Compute color values for each point
    color_values = normalized_dist_val
    # Reshape color_values to match the grid
    color_values_reshaped = color_values.reshape(x.shape)

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=color_values_reshaped, colorscale='Viridis')])

    fig.update_layout(title='density plot on a sphere',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'))

    fig.show()
 
 
 #vis for density maps
def batch_dist(x, t_pcl, sigma=1., R=1, batch_size=1024):
    bs = x.shape[:-1]
    xf = x.reshape(-1, 2)
    out = []
    dist_fn = jax.vmap(
        partial(weighted_dist, R=R, sigma=sigma),
        in_axes=(0, None)
    )
    for i in tqdm(range(0, xf.shape[0], batch_size)):
        j = min(i + batch_size, xf.shape[0])
        xb = xf[i:j]
        dist_val = dist_fn(xb, t_pcl)
        out.append(dist_val)
    return jnp.stack(out).reshape(*bs)

def gradient(x, t_pcl, sigma=1., batch_size=1024):
    bs = x.shape[:-1]
    xf = x.reshape(-1, 2)
    out = []
    grad_fun = jax.vmap(
        partial(weighted_grad, R=1, sigma=sigma),
        in_axes=(0, None)
    )
    for i in tqdm(range(0, xf.shape[0], batch_size)):
        j = min(i + batch_size, xf.shape[0])
        xb = xf[i:j]
        gradient_val = grad_fun(xb, t_pcl)
        gradient_norm = jnp.linalg.norm(gradient_val, axis=-1)
        out.append(gradient_norm)
    return jnp.stack(out).reshape(*bs)

def get_grid_index(x, y, resolution=256):
    grid_min = -2
    grid_max = 2

    # Calculate the size of each cell
    cell_size = (grid_max - grid_min) / resolution

    # Calculate the grid index for both x and y
    x_index = int((x - grid_min) / cell_size)
    y_index = int((y - grid_min) / cell_size)

    return x_index, y_index
           
    
    
    

def createPlane(normal, point_on_plane):
    normal = normal / np.linalg.norm(normal)
    
    # Find a vector in the plane
    if np.allclose(normal, [1, 0, 0]):
        v1 = np.cross(normal, [0, 1, 0])
    else:
        v1 = np.cross(normal, [1, 0, 0])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    half_width = 2
    half_height = 2
    
    # Calculate the corners
    corner1 = point_on_plane + half_width * v1 + half_height * v2
    corner2 = point_on_plane - half_width * v1 + half_height * v2
    corner3 = point_on_plane - half_width * v1 - half_height * v2
    corner4 = point_on_plane + half_width * v1 - half_height * v2

    vertices = np.array([corner1, corner2, corner3, corner4])

    # Define the faces of the rectangle
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [2, 1, 0],
        [3, 2, 0]
    ])

    # Create a mesh for the rectangle
    plane_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return plane_mesh

#for creating trajectory movies
def setup_plot(x_t, pcl, plot_hough=False):
    """Set up the initial 3D plot for multiple trajectories."""
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')
    ax.set_box_aspect([1,1,1])
    # Colors for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, x_t.shape[0]))

    # Initialize lines and points for each trajectory
    lines = [ax.plot([], [], [], '-', c=colors[i], alpha=0.1)[0] for i in range(x_t.shape[0])]
    pts = [ax.plot([], [], [], 'o', c=colors[i], markersize=2)[0] for i in range(x_t.shape[0])]
    if plot_hough:
        ax.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], c='gray', s=1.5, alpha=0.008)

    # Prepare the axes limits
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    # Set point-of-view
    ax.view_init(30, 0)
    return fig, ax, lines, pts

def capture_frame(x_t, i, fig, ax, lines, pts):
    """Capture a frame for the animation using precomputed trajectory data for multiple trajectories."""
    # Update line and point data for each trajectory
    for line, pt, traj in zip(lines, pts, x_t):
        if i > 0:  # Avoid errors on the first frame where i-1 would be invalid
            line.set_data(traj[:i, 0], traj[:i, 1])
            line.set_3d_properties(traj[:i, 2])
            pt.set_data(traj[i-1:i, 0], traj[i-1:i, 1])
            pt.set_3d_properties(traj[i-1:i, 2])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()

    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    return image

def return_frame(x_t, move_t, grid, ctrs, dist_val, valid):
    fig = plt.figure(figsize=(10,10))
    plt.contourf(grid[..., 0], grid[..., 1], dist_val * valid, levels=20)
    plt.scatter(x_t[:,0], x_t[:,1], c = 'r')
    for k in range(move_t.shape[0]):
        plt.arrow(x_t[k,0], x_t[k,1], 
                    dx=move_t[k,0], dy=move_t[k,1],
                    head_width=0.05)
    plt.scatter(ctrs[:, 0], ctrs[:, 1], c="r", s=1.0)
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    fig.canvas.draw()

    # Convert the canvas to a raw RGB buffer
    # buf = fig.canvas.tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[:, :, :3]
    plt.clf()
    plt.cla()
    plt.close()
    return image


#plotting lines/planes
def convert_to_pd(neural_pt, R=1): 
    """
    input
    nueral_pt: (2,)
    output:
    n, d: (2,), (1,)
    """
    norm = np.linalg.norm(neural_pt, axis=1)
    n = neural_pt / norm[:, None]
    d = norm - R

    theta = np.degrees(np.arctan2(n[:,1],n[:,0])) 
    return np.stack([theta, d], axis = 1) 


def parallel_line_through_point(theta, r):
    theta = np.radians(theta)
    (x, y) = (np.cos(theta) * r, np.sin(theta) * r)

    # turn x, y 90 degrees
    

    theta_prime = theta 
    (a, b) = (np.sin(theta_prime), -np.cos(theta_prime))
    return np.array([x, y]), np.array([a, b]), 

# def plot_point_direct_line(p, d, t0=-1e10, t1=1e100):
#    p0 = p + t0 * d 
#    p1 = p + t1 * d
#    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2, c ='r')

def plot_point_direct_line(p, d, t0=-1e10, t1=1e100, plot=plt, c='r'):
   p0 = p + t0 * d 
   p1 = p + t1 * d
   plt.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=3, c=c)