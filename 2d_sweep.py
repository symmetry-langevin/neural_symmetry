import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import cv2
from functools import partial
import torch
import wandb
import pickle
import os
from tqdm import tqdm
import argparse

from jax_utils import *
from vis_utils import *


def cross_product_batch_2d(p1, p2):
	return p1[...,0] * p2[...,1] - p1[...,1] * p2[...,0]

def hough_transform_batch_2d_change(p1, p2, eps1 = 1e-7, eps2 = 1e-2): #outputting degrees
    n = (p2 - p1) / torch.norm(p2 - p1, dim = -1, keepdim=True) # [...,2]
    m = (p1 + p2) / 2	# [...,2]
    d = torch.stack([n[...,1], -n[...,0]], dim = -1) # [...,2]
    r = -1 * cross_product_batch_2d(m, n) # [...]
    p3 = m + r.unsqueeze(-1) * d # [...,2]
    theta = torch.atan2(p3[...,1], p3[...,0]) # [...]
    # t = torch.sum(p3 *n, dim = -1)
    t = torch.norm(p3, dim = -1) # [...]
    nmask = (torch.norm(p1-p2, dim = -1)>eps2)
    return theta[nmask], t[nmask] # [...] and [...]

def batch_run(args):
    data_dir = args.data_path
    n_steps = args.n_steps
    sigma = args.sigma
    num_points = args.n_points
    step_size = args.step_size

    #log_name = f"{data_dir}/2d_langevin-{num_points}-{n_steps}-{step_size}-{sigma}"
    #wandb.init(project="2d_langevin_sweep", name=log_name, entity="xnf")

    files = os.listdir(data_dir)
    files.sort()
    #files = round_robin_sort(files)
    for file in files:
        file_path = os.path.join(data_dir, file)
        main(args, file_path)

def main(args, file_path):
    res   = args.res
    sigma = args.sigma
    R = args.R
    n_steps = args.n_steps
    num_points = args.n_points
    n_samples = args.n_samples
    step_size = args.step_size
    out_dir = args.out_path
    group_eps = args.group_eps
    group_min_points = args.group_min_points
    gen_vid = args.gen_vid
    
    data_name = '.'.join(file_path.split('/')[-1].split('.')[:-1])
    
    with open(file_path, 'rb') as output_file:
        ctrs = pickle.load(output_file)
    all_pts = jnp.stack([ctrs['x'], ctrs['y']], axis = 1)
    pts_norm = jnp.linalg.norm(all_pts, axis=-1).max()
    pts_center = all_pts.mean(axis=0).reshape(1, 2)
    ctrs = (all_pts - pts_center) / pts_norm

    
    samp_p1 = np.random.choice(np.arange(len(ctrs)), n_samples)
    samp_p2 = np.random.choice(np.arange(len(ctrs)), n_samples)
    p1 = torch.tensor(np.array(ctrs[samp_p1]))
    p2 = torch.tensor(np.array(ctrs[samp_p2]))

    rd  = hough_transform_batch_2d_change(p1, p2)
    theta = rd[0]
    d = rd[1][...,None]
    n = torch.stack([torch.cos(theta), torch.sin(theta)], dim = -1)
    d = (d + (R * torch.sign(d))) * n
    t_pcl = np.array(d)

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(t_pcl[:, 0], t_pcl[:, 1])
    ax1.axis("equal")

    grid = make_grid(res)
    dist_val = batch_dist(grid, t_pcl, sigma=sigma, R=R, batch_size=1024)
    valid = (jnp.linalg.norm(grid, axis=-1) > R).reshape(res, res)
    dist_val = dist_val.reshape(res, res)

    
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(ctrs[:, 0], ctrs[:, 1])
    ax2.contourf(grid[..., 0], grid[..., 1], dist_val, levels=10)
    ax2.axis("equal")
    
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    # Langevin
    def langevin_step(x, step_stats, R=1, sigma=0.03):
        t = step_stats["t"] 
        step_size = step_stats["step_size"]
        rng = jax.random.fold_in(key, t)
        z_t = jax.random.normal(rng, shape=x.shape)
        sch_ratio = jnp.sqrt(1 / (t + 1))
        # sch_ratio = 0

        noise_grad = jnp.sqrt(step_size) * z_t * sch_ratio
        x = jax.vmap(walk, in_axes=(0,0,None))(x, noise_grad, R)
        val = (jnp.linalg.norm(x, axis=-1) >= R)
        grad_val = jax.vmap(
            partial(weighted_grad, R=R, sigma=sigma),
            in_axes=(0, None)
        )(x, t_pcl)
        grad_val = jnp.where(~val[...,None], 0, grad_val)
        
        # (N, 2) , (N, 2) 
        move_t = - step_size * grad_val 
        x = jax.vmap(walk, in_axes=(0,0,None))(x, move_t, R)
        x_t = x
        return x, {"xt": x_t, "move_t": move_t}


    step_stats = {
        "t": jnp.arange(n_steps),
        "step_size": jnp.ones((n_steps,)) * step_size
    }

    x_init = jnp.array(t_pcl[np.random.choice(len(t_pcl), num_points, replace=False)])
    _, aux = jax.lax.scan(partial(langevin_step, R=R, sigma=sigma), x_init, step_stats)
    
    last = np.array(aux['xt'][-1])
    
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.contourf(grid[..., 0], grid[..., 1], dist_val, levels=10)
    ax3.scatter(ctrs[:, 0], ctrs[:, 1], c='r')
    ax3.scatter(last[:, 0], last[:, 1], c='b')
    ax3.axis("equal")


    #extract clusters
    my_labels = dbscan(last, eps=group_eps, MinPts=group_min_points)
    centroid = compute_centroids(last, my_labels)
    pt = np.array(centroid)
    converted = convert_to_pd(pt, R)
        
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.set_xlim([-2, 2])
    ax4.set_ylim([-2, 2])
    ax4.scatter(ctrs[:, 0], ctrs[:, 1], c='r', s=1)
    for i in range(len(converted)):
        theta, d = converted[i]
        p, d2 = parallel_line_through_point(theta, d)
        p0 = p + -2 * d2
        p1 = p + 2 * d2
        ax4.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2, c='b')


    # result saving
    fpath = f"{out_dir}/{data_name}_{step_size}_{sigma}.pickle"
    with open(fpath,'wb') as f:
        pickle.dump({"last": last, "centroid": centroid, "R": R, "theta": converted[:, 0], "r": converted[:, 1], "aux": aux}, f)
    
    # video logging 
    if gen_vid:
        step_bs = 400
        image_lst = []
        for i in tqdm(range(0, n_steps, step_bs)):
            frame_i = return_frame(aux["xt"][i], aux["move_t"][i], grid, ctrs, dist_val, valid)
            image_lst.append(frame_i)

        video_temp_path = "temp_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_temp_path, fourcc, 15, (image_lst[0].shape[0], image_lst[0].shape[1]))
        for im in image_lst:
            video_writer.write(im.astype('uint8'))
        video_writer.release()
        wandb.log({"video": wandb.Video(video_temp_path, fps=15, format="mp4"), 'transformation space' : wandb.Image(fig1), 'density': wandb.Image(fig2), 'last_step': wandb.Image(fig3), 'symmetries': wandb.Image(fig4)})
    else:
        wandb.log({'transformation space' : wandb.Image(fig1), 'density': wandb.Image(fig2), 'last_step': wandb.Image(fig3), 'symmetries': wandb.Image(fig4) })
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--R", type=float, default=0.3)
    parser.add_argument("--step_size", type=float, default=0.06)
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--n_samples", type=int, default=50_000)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--group_eps", type=int, default=0.05)
    parser.add_argument("--group_min_points", type=int, default=4)
    
    parser.add_argument("--data_path", type=str, default='./temp_test/shapes')
    parser.add_argument("--out_path", type=str, default= './temp_test/output')
    parser.add_argument("--gen_vid", type=bool, default=False)
    
    args = parser.parse_args()
    batch_run(args)