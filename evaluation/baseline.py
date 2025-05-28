'''
This file contains the functions to compress and uncompress a set of points using reflection lines.
Don't forget to normalize the points before using the compress function, such that the points are in the range [0, 1].
'''

from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import meshplot as mp
from torch_geometric.nn import knn
from sklearn import cluster
import sklearn



# 2D Utils

def hough_transform_with_normal(p1, p2, theta):
    theta = theta / 180 * np.pi
    n = np.array([np.cos(theta), np.sin(theta)])
    m = (p1 + p2) / 2
    c = np.dot(n, m)
    if theta < 0:
        theta += np.pi
        c *= -1
    return theta * 180 / np.pi, c

def hough_transform_batch_2d(p1, p2, eps1 = 1e-7): #outputting degrees
    def cross_product_batch_2d(p1, p2):
        return p1[...,0] * p2[...,1] - p1[...,1] * p2[...,0]
    n = (p2 - p1) / torch.norm(p2 - p1, dim = -1, keepdim=True) # [...,2]
    m = (p1 + p2) / 2	# [...,2]
    d = torch.stack([n[...,1], -n[...,0]], dim = -1) # [...,2]
    r = -1 * cross_product_batch_2d(m, n) # [...]
    p3 = m + r.unsqueeze(-1) * d # [...,2]
    theta = torch.atan2(p3[...,1], p3[...,0]) # [...]
    t = torch.sum(p3 *n, dim = -1)
    t = torch.norm(p3, dim = -1) # [...]
	# when symmetry line goes through the origin
    mask = (t < eps1) 
    t[mask] = 0
    theta[mask] = torch.arctan2(n[mask][...,1], n[mask][...,0])

    mask2 = theta < 0
    theta[mask2] += np.pi
    theta = theta * 180 / np.pi
    t[mask2] *= -1
    return theta, t # [...] and [...]

def reflection_line_from_hough(theta, r):
    theta = theta / 180 * np.pi
    (x, y) = (torch.cos(theta) * r, torch.sin(theta) * r)
    (a, b) = (torch.sin(theta), -torch.cos(theta))
    return torch.stack([x, y], dim=-1), torch.stack([a, b], dim=-1)

def reflection_line_from_hough_np(theta, r):
    theta = theta / 180 * np.pi
    (x, y) = (np.cos(theta) * r, np.sin(theta) * r)
    (a, b) = (np.sin(theta), -np.cos(theta))
    return np.column_stack([x, y]), np.column_stack([a, b])

def reflect_points(p, d, cts):
    d = d / np.linalg.norm(d)
    # Compute the vector from p to each point in cts
    c_minus_p = cts - p
    
    # Project c_minus_p onto d using dot product
    projection_scalar = np.dot(c_minus_p, d) / np.dot(d, d)
    projections = p + np.outer(projection_scalar, d)
    
    # Reflect the points across the line
    reflections = 2 * projections - cts
    
    return reflections

def reflect_points_multiple(p, d, q):
    # Normalize direction vectors
    d_normalized = d / d.norm(dim=1, keepdim=True)
    
    # Reshape q to broadcast over lines
    q_expanded = q.unsqueeze(0)  # shape: (1, n, 2)
    
    # Reshape p and d to broadcast over points
    p_expanded = p.unsqueeze(1)  # shape: (m, 1, 2)
    d_expanded = d_normalized.unsqueeze(1)  # shape: (m, 1, 2)

    # Compute pq_diff and the projection scalar
    pq_diff = q_expanded - p_expanded  # shape: (m, n, 2)
    pq_dot_d = torch.sum(pq_diff * d_expanded, dim=2, keepdim=True)  # shape: (m, n, 1)
    projection = pq_dot_d * d_expanded  # Projected component of q on d, shape: (m, n, 2)
    orthogonal_component = pq_diff - projection  # Orthogonal component to d, shape: (m, n, 2)

    # Reflect q across the line by subtracting twice the orthogonal component
    reflection = q_expanded - 2 * orthogonal_component  # shape: (m, n, 2)
    return reflection

def reflection_point_association(p, d, q, threshold):
    reflections = reflect_points_multiple(p, d, q)  # shape: (m, n, 2)
    reflections = reflections.view(-1, 2)  # Flatten to (m*n, 2)

    # Using knn to find the closest points in q for each point in reflections
    # knn finds indices of the nearest neighbors
    _, indices = knn(q, reflections, k=1) # indices shape: (m*n, 1)

    # Gather nearest points based on indices from q
    nearest_points = q[indices.squeeze()] # shape: (m*n, 2)

    # Calculate distances for the nearest neighbors
    distances = (nearest_points - reflections).norm(dim=1) # shape: (m*n,)

    # Reshape distances back to (m, n) and check threshold
    distances = distances.view(p.size(0), -1) # shape: (m, n)
    within_threshold = distances <= threshold # shape: (m, n)
    return within_threshold

def normalize(arr):
    diff_arr = np.max(arr) - np.min(arr)
    newarr = ((arr - np.min(arr))/diff_arr) -0.5

    return newarr

def denormalize(arr, oldarr):
    diff_arr = np.max(oldarr) - np.min(oldarr)
    newarr = (arr + 0.5) * diff_arr + np.min(oldarr)

    return newarr

def plot_point_direct_line(p, d, t0=-1e10, t1=1e100, plot=plt, c='r'):
   p0 = p + t0 * d 
   p1 = p + t1 * d
   plot.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=3, c=c)




# 2D Baseline

def compute_normals(points):
    _, idx = knn(points, points, k=3)

    idx_a = idx.view(-1, 3)[:, 1]
    idx_b = idx.view(-1, 3)[:, 2]

    a = points[idx_a]
    b = points[idx_b]

    neighbor_diff = a - b

    normals = torch.stack([neighbor_diff[:, 1], -neighbor_diff[:, 0]], dim=1)

    # Normalize normals
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return normals

def prune_points_normals(pairs_1_normals, pairs_2_normals, d, normal_difference_threadhold):
    d /= torch.norm(d, dim=1, keepdim=True)
    pairs_1_normals /= torch.norm(pairs_1_normals, dim=1, keepdim=True)
    pairs_2_normals /= torch.norm(pairs_2_normals, dim=1, keepdim=True)
    dot_products = torch.sum(pairs_1_normals * d, dim=1, keepdim=True)
    proj_on_uv = dot_products * d
    reflected_vectors = 2 * proj_on_uv - pairs_1_normals
    
    diff_norm1 = torch.norm(reflected_vectors - pairs_2_normals, dim=1)
    diff_norm2 = torch.norm(reflected_vectors + pairs_2_normals, dim=1)
    
    # Create a mask where either difference is within the threshold
    mask = (diff_norm1 < normal_difference_threadhold) | (diff_norm2 < normal_difference_threadhold)
    
    return mask

def get_basline_symmetries_gpu_with_normal_prunning(cts,
                             n_samples=100,
                             symmetry_point_min_distance = 0.1,
                             normal_max_dist=0.01):
    # sample n_samples points, create all paris and compute symmetry between them
    indices = torch.randperm(cts.shape[0])[:n_samples]
    all_pairs_indices = torch.combinations(indices, 2)
    pairs_1 = cts[all_pairs_indices[:, 0]]
    pairs_2 = cts[all_pairs_indices[:, 1]]
    theta, r = hough_transform_batch_2d(pairs_1, pairs_2)
    p, d = reflection_line_from_hough(theta, r)

    # pruninng based on normal and distance
    normals = compute_normals(cts)
    pairs_1_normals = normals[all_pairs_indices[:, 0]]
    pairs_2_normals = normals[all_pairs_indices[:, 1]]
    prunning1 = prune_points_normals(pairs_1_normals, pairs_2_normals, d, normal_max_dist)
    dist = torch.norm(pairs_1 - pairs_2, dim=1)
    prunning2 = dist > symmetry_point_min_distance
    prunning = prunning1 & prunning2

    pairs_1 = pairs_1[prunning]
    pairs_2 = pairs_2[prunning]
    theta = theta[prunning]
    r = r[prunning]
    p = p[prunning]
    d = d[prunning]

    returns = [(theta, r, p, d, pairs_1, pairs_2)]
    
    return returns

def get_baseline_symmetries_gpu(cts, 
                                n_samples=100,
                                symmetry_association_min_distance=0.01,
                                symmetry_ratio_threasholds=[0.3]):
    
    # assert n_samples > 1 and n_samples < cts.shape[0], "n_samples ill conditioned"
    indices = torch.randperm(cts.shape[0])[:n_samples]
    all_pairs_indices = torch.combinations(indices, 2)
    pairs_1 = cts[all_pairs_indices[:, 0]]
    pairs_2 = cts[all_pairs_indices[:, 1]]
        
    theta, r = hough_transform_batch_2d(pairs_1, pairs_2)
    p, d = reflection_line_from_hough(theta, r)
    association = reflection_point_association(p, d, cts, symmetry_association_min_distance)
    result_counts = association.sum(dim=1)

    returns = []
    for symmetry_ratio_threashold in symmetry_ratio_threasholds:
        mask = result_counts >= cts.shape[0] * symmetry_ratio_threashold
        returns.append((theta[mask], r[mask], p[mask], d[mask]))
    
    # return theta, r, p, d
    return returns

def get_baseline_symmetries_gpu_with_grouping(cts, 
                                n_samples=100,
                                symmetry_association_min_distance=0.01,
                                normal_max_dist=0.01,
                                symmetry_ratio_threashold=0.2,
                                grouping_theta_delta=10,
                                grouping_r_delta=0.05):
    
    returns = get_basline_symmetries_gpu_with_normal_prunning(cts, n_samples, symmetry_association_min_distance, normal_max_dist)
    theta, r, p, d, pairs_1, pairs_2 = returns[0]

    association = reflection_point_association(p, d, cts, symmetry_association_min_distance)
    result_counts = association.sum(dim=1)
    sorted_indices = torch.argsort(result_counts, descending=True)
    result_counts = result_counts[sorted_indices]
    p = p[sorted_indices]
    d = d[sorted_indices]
    theta = theta[sorted_indices]
    r = r[sorted_indices]
    pairs_1 = pairs_1[sorted_indices]
    pairs_2 = pairs_2[sorted_indices]

    mask = result_counts >= cts.shape[0] * symmetry_ratio_threashold
    theta = theta[mask]
    r = r[mask]
    p = p[mask]
    d = d[mask]
    pairs_1 = pairs_1[mask]
    pairs_2 = pairs_2[mask]

    final_return_indices = []

    for i in range(theta.shape[0]):
        include = True
        for j in range(len(final_return_indices)):
            theta_1 = theta[i]
            r_1 = r[i]
            theta_2 = theta_1 + 180
            r_2 = -r_1
            theta_3 = theta_1 - 180
            r_3 = -r_1
            theta_4 = theta_1
            r_4 = -r_1 if (theta_1 > -1e-5 and theta_1 < 1e-5) else r_1
            theta_j = theta[final_return_indices[j]]
            r_j = r[final_return_indices[j]]
            if (abs(theta_1 - theta_j) < grouping_theta_delta and abs(r_1 - r_j) < grouping_r_delta) or (abs(theta_2 - theta_j) < grouping_theta_delta and abs(r_2 - r_j) < grouping_r_delta) or (abs(theta_3 - theta_j) < grouping_theta_delta and abs(r_3 - r_j) < grouping_r_delta) or (abs(theta_4 - theta_j) < grouping_theta_delta and abs(r_4 - r_j) < grouping_r_delta):
                include = False
                break

        if include:
            final_return_indices.append(i)
    
    theta = theta[final_return_indices]
    r = r[final_return_indices]
    p = p[final_return_indices]
    d = d[final_return_indices]
    pairs_1 = pairs_1[final_return_indices]
    pairs_2 = pairs_2[final_return_indices]

    return (theta, r, p, d, pairs_1, pairs_2)

def filter_symmetries(theta,
                      r,
                      grouping_theta_delta=10,
                      grouping_r_delta=0.05):
    
    p, d = reflection_line_from_hough(theta, r)
    final_return_indices = []
    for i in range(theta.shape[0]):
        include = True
        for j in range(len(final_return_indices)):
            theta_1 = theta[i]
            r_1 = r[i]
            theta_2 = theta_1 + 180
            r_2 = -r_1
            theta_3 = theta_1 - 180
            r_3 = -r_1
            theta_4 = theta_1
            r_4 = -r_1 if (theta_1 > -1e-5 and theta_1 < 1e-5) else r_1
            theta_j = theta[final_return_indices[j]]
            r_j = r[final_return_indices[j]]
            if (abs(theta_1 - theta_j) < grouping_theta_delta and abs(r_1 - r_j) < grouping_r_delta) or (abs(theta_2 - theta_j) < grouping_theta_delta and abs(r_2 - r_j) < grouping_r_delta) or (abs(theta_3 - theta_j) < grouping_theta_delta and abs(r_3 - r_j) < grouping_r_delta) or (abs(theta_4 - theta_j) < grouping_theta_delta and abs(r_4 - r_j) < grouping_r_delta):
                include = False
                break

        if include:
            final_return_indices.append(i)
    
    theta = theta[final_return_indices]
    r = r[final_return_indices]
    p = p[final_return_indices]
    d = d[final_return_indices]

    return (theta, r, p, d)


def patch_growing(cts, 
                  theta, 
                  r, 
                  p, 
                  d, 
                  pairs_1, 
                  pairs_2, 
                  neighbor_threshold=0.06, 
                  max_distance=0.02, 
                  num_modes_to_grow=5, # None for all
                  quantile=0.1,
                  symmetry_ratio_threashold=0.05,
                  plot=False):
    
    theta_normalized = normalize(theta)
    r_normalized = normalize(r)
    transformation_space_points = np.column_stack((theta_normalized, r_normalized))
    bandwidth = cluster.estimate_bandwidth(transformation_space_points, quantile=quantile)
    ms = cluster.MeanShift(bandwidth=bandwidth)
    ms.fit(transformation_space_points)
    cluster_centers = ms.cluster_centers_ # modes

    tree = sklearn.neighbors.KDTree(transformation_space_points)
    _, indices = tree.query(cluster_centers, k=1)

    if num_modes_to_grow == None:
        num_modes_to_grow = indices.shape[0]

    patch_growing_result = []

    for i in range(num_modes_to_grow):
        p1 = pairs_1[indices[i]].flatten().tolist()
        p2 = pairs_2[indices[i]].flatten().tolist()
        p_single = p[indices[i]].flatten()
        d_single = d[indices[i]].flatten()
        theta_single = theta[indices[i]].flatten()
        r_single = r[indices[i]].flatten()

        cts_list = cts.tolist()
        tree = sklearn.neighbors.KDTree(cts_list)
        visited = {cts_list.index(p1), cts_list.index(p2)}
        left_patch = [p1]
        right_patch = [p2]

        queue = [(p1, p2)]
        while(len(queue) > 0):
            p1, p2 = queue.pop(0)
            neighbor_indices = tree.query_radius([p1], neighbor_threshold)[0].tolist()

            for neighbor_index in neighbor_indices:
                if neighbor_index in visited:
                    continue
                neighbor = cts[neighbor_index]
                reflected = reflect_points(p_single, d_single, neighbor)[0]
                neighbor = neighbor.flatten().tolist()
                reflected_neighbor_indices = tree.query_radius([reflected], max_distance)[0].tolist()

                # find first index that is not in visited
                nearest_reflected_index = None
                for reflected_neighbor_index in reflected_neighbor_indices:
                    if reflected_neighbor_index not in visited:
                        nearest_reflected_index = reflected_neighbor_index
                        break

                if nearest_reflected_index == None:
                    continue
                visited.add(neighbor_index)
                visited.add(nearest_reflected_index)
                nearest_reflected =  cts[nearest_reflected_index].flatten().tolist()
                queue.append((neighbor, nearest_reflected))
                left_patch.append(neighbor)
                right_patch.append(nearest_reflected)

        left_patch = np.array(left_patch)
        right_patch = np.array(right_patch)
        if left_patch.shape[0] < cts.shape[0] * symmetry_ratio_threashold:
            continue

        patch_growing_result.append((left_patch, right_patch, p_single, d_single, theta_single, r_single))

        if plot:
            plt.scatter(left_patch[:,0], left_patch[:,1], c='purple')
            plt.scatter(right_patch[:,0], right_patch[:,1], c='orange')
            plot_point_direct_line(p_single, d_single, plot=plt)

            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()
    
    return patch_growing_result




# 3D Utils

def hough_transform_3d_1(p1, p2):
    n = (p2 - p1) / torch.norm(p2 - p1, dim=-1, keepdim=True)  # [...,3]
    m = (p1 + p2) / 2  # [...,3]
    d = torch.sum(m * n, dim=-1) # dot product project m onto n

    # flip n 180 degrees if d is negative TODO: check if this is correct
    flip_indices = d < 0
    n[flip_indices] = -n[flip_indices]
    d = torch.abs(d)

    # point on the plane
    p = n * d.unsqueeze(-1)  # [...,3]

    return d, n, p, m # distance, normal, point on the plane, middle point between points. Normal is already normalized

def hough_transform_3d_2(p1, p2):
    n = (p2 - p1) / torch.norm(p2 - p1, dim=-1, keepdim=True)  # [...,3]
    m = (p1 + p2) / 2  # [...,3]
    d = torch.sum(m * n, dim=-1) # dot product project m onto n

    # randomly sample half of the points to flip the normal
    flip_indices = torch.rand(d.shape) > 0.5
    n[flip_indices] = -n[flip_indices]
    d[flip_indices] = -d[flip_indices]

    p = n * d.unsqueeze(-1)  # [...,3]

    return d, n, p, m # distance, normal, point on the plane, middle point between points. Normal is already normalized

def reflect_point_3d(plane, ct):
    a, b, c, d = plane

    # Calculate the projection of ct onto the plane
    t = (a * ct[0] + b * ct[1] + c * ct[2] + d) / (a**2 + b**2 + c**2)
    projection = ct - t * np.array([a, b, c])

    # Calculate the reflection of ct across the plane
    reflection = 2 * projection - ct

    return reflection

def reflect_points_3d(plane, cts):
    a, b, c, d = plane

    # Calculate the projection of ct onto the plane
    t = (a * cts[:,0] + b * cts[:,1] + c * cts[:,2] + d) / (a**2 + b**2 + c**2)
    projections = cts - np.outer(t, np.array([a, b, c]))

    # Calculate the reflection of ct across the plane
    reflections = 2 * projections - cts

    return reflections

def reflect_points_multiple_3d(d, n, p):
    points_expanded = p.unsqueeze(0).expand(n.size(0), -1, -1)
    normals_expanded = n.unsqueeze(1).expand(-1, p.size(0), -1)
    distances_expanded = d.unsqueeze(1).expand(-1, p.size(0))
    dot_products = torch.sum(points_expanded * normals_expanded, dim=2)
    # reflections: (m, n, 3)
    reflections = points_expanded - 2 * (dot_products - distances_expanded).unsqueeze(2) * normals_expanded

    return reflections

def get_reflection_infos_3d(cts, reflection_planes, delta=0.1):
    kdtree = KDTree(cts)
    reflection_points = [[] for _ in range(len(reflection_planes))]

    for i in range(len(reflection_planes)):
        plane = reflection_planes[i]
        for ct in cts:
            reflection = reflect_point_3d(plane, ct)
            distance, index = kdtree.query(reflection)
            if distance < delta:
                reflection_points[i].append(ct)

        reflection_points[i] = np.array(reflection_points[i])

    return [[reflection_planes[i], np.array(reflection_points[i])] for i in range(len(reflection_planes))]

def reflection_point_association_3d(d, n, q, threshold):
    # d in shape (m, ), n in shape (m, 3), q in shape (n, 3)
    reflections = reflect_points_multiple_3d(d, n, q)  # shape: (m, n, 3)
    reflections = reflections.view(-1, 3)  # Flatten to (m*n, 3)

    # Using knn to find the closest points in q for each point in reflections
    # knn finds indices of the nearest neighbors
    _, indices = knn(q, reflections, k=1, batch_x=None, batch_y=None) # indices shape: (m*n, 1)

    # Gather nearest points based on indices from q
    nearest_points = q[indices.squeeze()] # shape: (m*n, 3)

    # Calculate distances for the nearest neighbors
    distances = (nearest_points - reflections).norm(dim=1) # shape: (m*n,)

    # Reshape distances back to (m, n) and check threshold
    distances = distances.view(d.size(0), -1) # shape: (m, n)
    within_threshold = distances <= threshold # shape: (m, n), sum this along axis 1 to get the number of associated points
    return within_threshold


# 3D Baseline

def compute_normals_3d(points, k=10):
    knn_idx = knn(points, points, k=k)  # Placeholder for the knn function
    knn_idx = knn_idx[1, :].reshape(-1, k)  # shape [num_points, k]
    knn_points = points[knn_idx]  # shape [num_points, k, 3]
    centroid = knn_points.mean(dim=1, keepdim=True)
    centered_points = knn_points - centroid  # shape [num_points, k, 3]
    cov_matrix = torch.matmul(centered_points.transpose(1, 2), centered_points) / (k - 1)
    _, eigvec = torch.linalg.eigh(cov_matrix)  # eigvec shape [num_points, 3, 3]
    normal = eigvec[:, :, 0]  # Take the first column for each point
    # normal is already normalized
    return normal

def prune_points_normals_3d(pairs_1_normals, pairs_2_normals, d, normal_difference_threadhold):
    dot_products = torch.sum(pairs_1_normals * d, dim=1, keepdim=True)
    proj_on_uv = dot_products * d
    reflected_vectors = 2 * proj_on_uv - pairs_1_normals
    
    diff_norm1 = torch.norm(reflected_vectors - pairs_2_normals, dim=1)
    diff_norm2 = torch.norm(reflected_vectors + pairs_2_normals, dim=1)
    
    # Create a mask where either difference is within the threshold
    mask = (diff_norm1 < normal_difference_threadhold) | (diff_norm2 < normal_difference_threadhold)
    
    return mask

def get_basline_symmetries_gpu_with_normal_prunning_3d(points,
                                 n_samples=1000000,
                                 symmetry_point_min_distance=0.1,
                                 normal_max_dist=0.01):
    point_normals = compute_normals_3d(points)
	
	# sample 100k points with replacement
    sample_index_1 = torch.randint(0, points.shape[0], (n_samples,))
    sample_index_2 = torch.randint(0, points.shape[0], (n_samples,))
    sampled_points_1 = points[sample_index_1]
    sampled_point_normals_1 = point_normals[sample_index_1]
    sampled_points_2 = points[sample_index_2]
    sampled_point_normals_2 = point_normals[sample_index_2]

    d, n, plane_p, m = hough_transform_3d_1(sampled_points_1, sampled_points_2)

    prunning1 = prune_points_normals_3d(sampled_point_normals_1, sampled_point_normals_2, n, normal_max_dist)
    dist = torch.norm(sampled_points_1 - sampled_points_2, dim=1)
    prunning2 = dist > symmetry_point_min_distance
    prunning = prunning1 & prunning2

    sampled_points_1 = sampled_points_1[prunning]
    sampled_points_2 = sampled_points_2[prunning]
    d = d[prunning]
    n = n[prunning]
    plane_p = plane_p[prunning]
    m = m[prunning]

    return sampled_points_1, sampled_points_2, d, n, plane_p, m

def get_baseline_symmetries_gpu_with_grouping_3d(cts, 
                                n_samples=1000000,
                                symmetry_association_min_distance=0.01,
                                normal_max_dist=0.01,
                                symmetry_point_min_distance = 0.1,
                                symmetry_ratio_threashold=0.2,
                                grouping_transformation_space_delta=0.1):
    
    returns = get_basline_symmetries_gpu_with_normal_prunning_3d(cts, n_samples, symmetry_point_min_distance, normal_max_dist)
    sampled_points_1, sampled_points_2, d, n, plane_p, m = returns

    association = reflection_point_association_3d(d, n, cts, symmetry_association_min_distance)
    result_counts = association.sum(dim=1)
    sorted_indices = torch.argsort(result_counts, descending=True)
    result_counts = result_counts[sorted_indices]
    d = d[sorted_indices]
    n = n[sorted_indices]
    plane_p = plane_p[sorted_indices]
    m = m[sorted_indices]
    sampled_points_1 = sampled_points_1[sorted_indices]
    sampled_points_2 = sampled_points_2[sorted_indices]

    mask = result_counts >= cts.shape[0] * symmetry_ratio_threashold
    d = d[mask]
    n = n[mask]
    plane_p = plane_p[mask]
    m = m[mask]
    sampled_points_1 = sampled_points_1[mask]
    sampled_points_2 = sampled_points_2[mask]

    final_return_indices = []

    for i in range(d.shape[0]):
        include = True
        for j in range(len(final_return_indices)):
            d_single = d[i]
            n_single = n[i]
            transformation_space_point_1 = n_single * (d_single + torch.sign(d_single))
            trasnformation_space_point_2 = n_single * (- d_single - torch.sign(d_single))
            
            d_j = d[final_return_indices[j]]
            n_j = n[final_return_indices[j]]
            transformation_space_point_j = n_j * (d_j + torch.sign(d_j))
            if torch.norm(transformation_space_point_1 - transformation_space_point_j) < grouping_transformation_space_delta or torch.norm(trasnformation_space_point_2 - transformation_space_point_j) < grouping_transformation_space_delta:
                include = False
                break

        if include:
            final_return_indices.append(i)
    
    d = d[final_return_indices]
    n = n[final_return_indices]
    plane_p = plane_p[final_return_indices]
    m = m[final_return_indices]
    sampled_points_1 = sampled_points_1[final_return_indices]
    sampled_points_2 = sampled_points_2[final_return_indices]

    return (sampled_points_1, sampled_points_2, d, n, plane_p, m)


# Compression

# p, d are pytorch tensor of size (2, ) and (2, ) denoting the points and directions of the reflection lines, and cts is a pytorch tensor of size (n, 2) denoting the points to be reflected
def split_points_by_line(p, d, cts):
    # Calculate the normal vector to the line
    n = torch.tensor([-d[1], d[0]]).to(d.device)
    
    # Calculate the vectors from p to each point in cts
    vectors = cts - p
    
    # Dot product of each vector with the normal vector
    dot_products = torch.sum(vectors * n, dim=1)
    
    # Split the points based on the sign of the dot product
    side1 = cts[dot_products >= 0]
    side2 = cts[dot_products <= 0]
    
    return side1, side2

# p, d torch tensor of shape (m, 2). points torch tensor of shape (n, 2)
def compress(p, d, points, delta, plot=False):
    if points.shape[0] < 10:
        return [points.cpu().numpy()]
    
    association = reflection_point_association(p, d, points, delta) # shape: (m, n)

     # Count the number of true values for each line
    reflection_count = association.sum(dim=1) # shape: (m,)

    # get row of highest count and that count
    max_count = torch.max(reflection_count)
    if(max_count < 10):
        return [points.cpu().numpy()]
    
    max_count_index = torch.argmax(reflection_count)
    best_p = p[max_count_index]
    best_d = d[max_count_index]
    remaining_mask = torch.ones(p.shape[0], dtype=torch.bool)
    remaining_mask[max_count_index] = False
    remaining_p = p[remaining_mask]
    remaining_d = d[remaining_mask]

    if remaining_p.dim() == 1:
        remaining_p = remaining_p.unsqueeze(0)
        remaining_d = remaining_d.unsqueeze(0)

    points_associated = points[association[max_count_index]]
    points_not_associated = points[~association[max_count_index]]

    if points_associated.dim() == 1:
        points_associated = points_associated.unsqueeze(0)
        points_not_associated = points_not_associated.unsqueeze(0)

    A, B = split_points_by_line(best_p, best_d, points_associated)

    if plot:
        A_numpy = A.cpu().numpy()
        B_numpy = B.cpu().numpy()
        remaining = points_not_associated.cpu().numpy()
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.scatter(A_numpy[:,0], A_numpy[:,1], c='b')
        plt.scatter(B_numpy[:,0], B_numpy[:,1], c='g')
        plt.scatter(remaining[:,0], remaining[:,1], c='r')
        plot_point_direct_line(best_p.cpu().numpy(), best_d.cpu().numpy())
        plt.show()


    if(p.shape[0] == 1):
        return [best_p.cpu().numpy(), best_d.cpu().numpy(), [A.cpu().numpy() if A.shape[0] < B.shape[0] else B.cpu().numpy()], [points_not_associated.cpu().numpy()]]

    association_A = reflection_point_association(remaining_p, remaining_d, A, delta)
    association_B = reflection_point_association(remaining_p, remaining_d, B, delta)

    # heuristic 1
    sum_association_A = torch.sum(association_A)
    sum_association_B = torch.sum(association_B)
    saved = A if sum_association_A > sum_association_B else B

    return [best_p.cpu().numpy(), best_d.cpu().numpy(), compress(remaining_p, remaining_d, saved, delta, plot=plot), compress(remaining_p, remaining_d, points_not_associated, delta, plot=plot)]

'''
Decompress the compressed points
Parameters:
	compressed: compressed object from the compress function
'''
def decompress(compressed):
    if len(compressed) == 1:
        return compressed[0]

    A = decompress(compressed[2])
    B = reflect_points(compressed[0], compressed[1], A)
    remaining = decompress(compressed[3])

    return np.concatenate((A, B, remaining))

'''
Calculate the number of points in the compressed object
Parameters:
	compressed: compressed object from the compress function
'''
def points_memorized(compressed):
    if len(compressed) == 1:
        return compressed[0].shape[0]

    return points_memorized(compressed[2]) + points_memorized(compressed[3])

'''
Calculate the space used by the compressed object
Parameters:
	compressed: compressed object from the compress function
'''
def space_used(compressed):
    if len(compressed) == 1:
        return compressed[0].nbytes

    return compressed[0].nbytes + compressed[1].nbytes + space_used(compressed[2]) + space_used(compressed[3])

'''
Visualize the compressed object
Parameters:
	compressed: compressed object from the compress function
'''
def visualize_compressed(compressed):
    if len(compressed) == 1:
        plt.scatter(compressed[0][:,0], compressed[0][:,1])
        return

    visualize_compressed(compressed[2])
    visualize_compressed(compressed[3])
    # plot_point_direct_line(compressed[0], compressed[1])
    
'''
Calculate the square root of the mean squared error of the compression
Parameters:
	compressed: compressed object from the compress function
	uncompressed: numpy array of points
'''
def compression_sqrt_mse(cts, uncompressed):
	tree = KDTree(cts)
	sum = 0
	for point in uncompressed:
		distance, index = tree.query(point)
		sum += distance**2

	sum /= uncompressed.shape[0]
	return np.sqrt(sum)



# 3D Compression

def split_points_by_plane(plane, cts):
    a, b, c, d = plane

    # Calculate the dot products
    dot_products = a * cts[:,0] + b * cts[:,1] + c * cts[:,2] + d

    # Split the points based on the sign of the dot product
    side1 = cts[dot_products >= 0]
    side2 = cts[dot_products <= 0]

    return side1, side2

def compress_3d(reflection_planes, points, delta):
    if points.shape[0] < 10:
        return [None, None, points]

    reflection_all = get_reflection_infos_3d(points, reflection_planes, delta=delta)
    reflection_all = sorted(reflection_all, key=lambda x: x[1].shape[0], reverse=True)

    current_reflection = reflection_all[0]

    if current_reflection[1].shape[0] < 10:
        return [None, None, points]

    plane = current_reflection[0]
    reflection_point_select = current_reflection[1]
    A, B = split_points_by_plane(plane, reflection_point_select)
    remaining = points[~np.isin(points, reflection_point_select).all(1)]


    plt = mp.plot(A, shading={"point_color": "blue", "width": 200, "height": 200, "point_size": 0.02})
    plt.add_points(B, shading={"point_color": "green", "point_size": 0.02})
    plt.add_points(remaining, shading={"point_color": "red", "point_size": 0.02})

    if(len(reflection_planes) == 1):
        return [plane, (None, None, A), (None, None, remaining)]

    remaining_reflections = reflection_planes[1:]

    return [plane, compress_3d(remaining_reflections, A, delta), compress_3d(remaining_reflections, remaining, delta)]

def uncompress_3d(compressed):
    if compressed[0] is None:
        return compressed[2]

    A = uncompress_3d(compressed[1])
    B = reflect_points_3d(compressed[0], A)
    remaining = uncompress_3d(compressed[2])

    return np.concatenate((A, B, remaining))

def points_memorized_3d(compressed):
    if compressed[0] is None:
        return compressed[2]

    return np.concatenate((points_memorized_3d(compressed[1]), points_memorized_3d(compressed[2])))

def space_used_3d(compressed):
    if compressed[0] is None:
        return (compressed[2].size * compressed[2].itemsize) + sys.getsizeof(compressed)

    return sys.getsizeof(compressed) + space_used_3d(compressed[1]) + space_used_3d(compressed[2])

def compression_sqrt_mse_3d(cts, uncompressed):
    tree = KDTree(cts)
    sum = 0
    for point in uncompressed:
        distance, index = tree.query(point)
        sum += distance**2

    sum /= uncompressed.shape[0]
    return np.sqrt(sum)

