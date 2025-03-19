import jax.numpy as jnp
import numpy as np
import time

#all for vis
import open3d as o3d
import wandb
import pickle
import argparse

from jax_utils import *
from jax_utils_3d import *
from vis_utils import *

def hough(x, y):
   n = (x-y) / jnp.linalg.norm(x - y)
   m = (x+y) / 2
   d = jnp.dot(m, n)
   return n, d
hough_vec = jax.vmap(hough, in_axes=(0, 0))

def main(args):

    #plot_all_3d([to_plot])
    data_name = args.data_path.split('/')[-1].split('.')[0]

    
    R = 1
    num_samples = args.n_samples
    num_points = args.n_points
    n_steps = args.n_steps
    sigma = args.sigma
    step_stats = {
        "t": jnp.arange(n_steps),
        "sigma": jnp.ones((n_steps,)) * sigma
    }
    
    #log_name = f"{data_name}/{num_samples}-{sigma}"
    #wandb.init(project="symmetry-sweep", name=log_name, entity="xnf")
    
    
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    mesh = o3d.io.read_triangle_mesh(args.data_path)
    center_of_mass = mesh.get_center()
    mesh.translate(-center_of_mass)
    norms = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    mesh.scale(1/np.max(norms), center=(0,0,0))

    p1 = jnp.array(mesh.sample_points_uniformly(num_samples).points)
    p2 = jnp.array(mesh.sample_points_uniformly(num_samples).points)

    ns, ds = hough_vec(p1, p2)
    n_times_d = ns * (ds + np.sign(ds))[:, np.newaxis] #(n_samples, 3)  
    n_times_d= jnp.array(n_times_d)
    #to_plot = create_points_3d(n_times_d)

    #wandb.log({"hough space": plot_all_3d([to_plot])})
    #wandb.log({"density": vis_density(100, n_times_d, sigma=sigma, bs=100) })
    #save_pc('hough space', n_times_d)

    def langevin_step(x, step_stats):
        t = step_stats["t"] 
        sigma = step_stats["sigma"]
        step_size = 2 * sigma ** 2 
        step_size = step_size * 10 # tune
        rng = jax.random.fold_in(key, t)
        z_t = jax.random.normal(rng, shape=x.shape)
        sch_ratio = jnp.sqrt(1 / (t + 1))
        noise_grad = jnp.sqrt(step_size) * z_t * sch_ratio
        x = jax.vmap(walk_3d, in_axes=(0,0))(x, noise_grad)
        #print(x.shape)
        val = jnp.linalg.norm(x, axis=-1) >= R
        
        #breakpoint_if_nonfinite(val)
        grad_val = jax.vmap(
            partial(weighted_grad_3d, R=R, sigma=sigma),
            in_axes=(0, None)
        )(x, n_times_d)
        
        grad_val = jnp.where(~val[...,None], 0, grad_val)
        # (N, 2) , (N, 2) 
        move_t = -step_size * grad_val 
        x = jax.vmap(walk_3d, in_axes=(0,0))(x, move_t)
        x_t = x
        return x, {"xt": x_t, "move_t": move_t}


    rng = jax.random.PRNGKey(int(time.time()))
    rng1, rng2 = jax.random.split(rng)
    x = jax.random.normal(rng1, shape=(num_points, 3))
    xd = x / safe_norm(x)
    xr = jax.random.uniform(rng2, shape=(num_points, 1)) * 0.5
    # xr = 0
    x_init = (xr + 1) * xd

    #x_endpoint, aux = batch_langevin_steps(x_init, step_stats, rng, batch_size=30)

    # x_init = jnp.array([[1.5, 1.5]])
    _, aux = jax.lax.scan(langevin_step, x_init, step_stats)
    xt = aux['xt']
    #index = jnp.linspace(0, len(xt)-1, 100, dtype=int)
    #x_t = jnp.transpose(jnp.array(xt)[index], (1, 0, 2))
    
    last = xt[-1]
    #my_labels = dbscan(last, eps=0.1, MinPts=2) #TODO:this has to be tuned!
    #centroid = compute_centroids(last, my_labels)
    #centroids = np.array(centroid)
    #print("total number of modes found: " + str(len(centroids)))
    
    #all_points = create_points_3d(p1, alpha = 0.05, label = 'primal')
    #hough = create_points_3d(n_times_d, alpha = 0.05, label = 'dual')
    #start = create_points_3d(xt[0], alpha = 1, label = 'initial timestep')
    #end = create_points_3d(xt[-1], alpha = 1, markersize = 4, label = 'final timestep')

    
    #wandb.log({"langevin": plot_all_3d([hough, start, end], grid = False)})
    fpath = f"{args.out_path}/{data_name}_{num_samples}_{sigma}_modes.pickle"
    with open(fpath,'wb') as f:
        pickle.dump(last, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--n_points", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--out_path", type=str, default='./temp_3d/')
    parser.add_argument("--data_path", type=str, default='./temp_data/')
    args = parser.parse_args()
    main(args)
