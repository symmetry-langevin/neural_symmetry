import time
import argparse

import open3d as o3d
import pickle

import jax
import jax.numpy as jnp
from safe_jax_utils import *
from jax_utils_3d import *


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
    
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    #####
    #LOADS POINTS  -> needs to be preprocessed
    mesh = o3d.io.read_triangle_mesh(args.data_path)
    center_of_mass = mesh.get_center()
    mesh.translate(-center_of_mass)
    norms = jnp.linalg.norm(jnp.array(mesh.vertices), axis=1)
    mesh.scale(1/jnp.max(norms), center=(0,0,0))

    p1 = jnp.array(mesh.sample_points_uniformly(num_samples).points)
    p2 = jnp.array(mesh.sample_points_uniformly(num_samples).points)
    #### all this could be predumped
    
    ns, ds = hough_vec(p1, p2)
    n_times_d = ns * (ds + jnp.sign(ds))[:, jnp.newaxis] #(n_samples, 3)  
    n_times_d= jnp.array(n_times_d)


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
    x_init = (xr + 1) * xd


    _, aux = jax.lax.scan(langevin_step, x_init, step_stats)
    xt = aux['xt']
    last = xt[-1]
    #dumping the outputs
    fpath = f"{args.out_path}/{data_name}_{num_samples}_{sigma}_modes.pickle"
    with open(fpath,'wb') as f:
        pickle.dump(last, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100_000)
    parser.add_argument("--n_steps", type=int, default=10_000)
    parser.add_argument("--n_points", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--out_path", type=str, default='./temp_3d/')
    parser.add_argument("--data_path", type=str, default='./temp_data/')
    args = parser.parse_args()
    main(args)
