import torch
import matplotlib.pyplot as plt
import numpy as np
from baseline import *
import os
import pickle
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Torch version:', torch.__version__)
print('Using device:', device)

shapes_directory = 'sample_set'
groundtruth_directory = 'output/symmetries'
method_directory = './output/mitra'
method_output_files = os.listdir(method_directory)

grouping_theta_delta = 20
grouping_r_delta = 0.3


def eval(file_name):
	# print(file_name)
	with open(os.path.join(method_directory, file_name), 'rb') as output_file:
		symmetry = pickle.load(output_file)
	theta = symmetry['theta']
	r = symmetry['r']

	theta, r, p, d = filter_symmetries(torch.tensor(theta, device=device), torch.tensor(r, device=device), grouping_theta_delta, grouping_r_delta)
	theta = theta.cpu().numpy()
	r = r.cpu().numpy()
	p = p.reshape(-1, 2)
	d = d.reshape(-1, 2)

	shape_name = file_name.split('.')[0].split('-')[0]
	groundtruth_file_name = shape_name + "_groundtruth.pkl"
	shape_file_name = shape_name + ".pickle"

	with open(os.path.join(shapes_directory, shape_file_name), 'rb') as output_file:
		ctrs = pickle.load(output_file)
	cts = np.stack([ctrs['x'], ctrs['y']], axis = 1, dtype=np.float64)
	cts = cts - np.mean(cts, axis=0)
	cts = cts / np.max(np.linalg.norm(cts, axis=1))
	cts_tensor = torch.tensor(cts, dtype=torch.float64, device=device, requires_grad=False)

	with open(os.path.join(groundtruth_directory, groundtruth_file_name), 'rb') as output_file:
		symmetry = pickle.load(output_file)
	theta_gt = symmetry['theta']
	r_gt = symmetry['r']

	detected = theta.shape[0]
	total = theta_gt.shape[0]
	true_positive = 0


	for i in range(detected):
		positive = False
		for j in range(total):
			theta_1 = theta[i]
			r_1 = r[i]
			theta_2 = theta_1 + 180
			r_2 = -r_1
			theta_3 = theta_1 - 180
			r_3 = -r_1
			theta_4 = theta_1
			r_4 = -r_1 if (theta_1 > -1e-5 and theta_1 < 1e-5) else r_1
			theta_j = theta_gt[j]
			r_j = r_gt[j]
			if (abs(theta_1 - theta_j) < grouping_theta_delta and abs(r_1 - r_j) < grouping_r_delta) or (abs(theta_2 - theta_j) < grouping_theta_delta and abs(r_2 - r_j) < grouping_r_delta) or (abs(theta_3 - theta_j) < grouping_theta_delta and abs(r_3 - r_j) < grouping_r_delta) or (abs(theta_4 - theta_j) < grouping_theta_delta and abs(r_4 - r_j) < grouping_r_delta):
				positive = True
				break
		if positive:
			true_positive += 1
	
	if detected == 0:
		precision = 0
		recall = 0
		compression_ratio = 1
		association_measure = 0
	else:
		precision = true_positive / detected
		recall = true_positive / total

		# Compression
		compressed = compress(p, d, cts_tensor, 0.03, plot=False)
		original_space_used = cts.nbytes
		compressed_space_used = space_used(compressed)
		compression_ratio = compressed_space_used / original_space_used

		# Assocation Measure
		association = reflection_point_association(p, d, cts_tensor, 0.03)
		result_counts = association.sum(dim=1)
		thresholds = np.linspace(0, 1, 100)
		scores = []
		for threshold in thresholds:
			scores.append((result_counts > threshold * cts_tensor.shape[0]).sum() / result_counts.shape[0])
		association_measure = np.trapz(scores, thresholds) # area under curve
	
	return precision, recall, compression_ratio, association_measure

def eval_noise_level(noise_level):
	
	precisions = []
	recalls = []
	compression_ratios = []
	association_measures = []
	# file_include_str = '-'+str(noise_level)+'.p'
	file_include_str = '-'+str(noise_level)+'_'
	noise_level_files = [file for file in method_output_files if file_include_str in file]

	if len(noise_level_files) == 0:
		print('Noise level:', noise_level, ', number of files:', len(noise_level_files), ', No files found')
		return

	for method_output_file in noise_level_files:
		precision, recall, compression_ratio, association_measure = eval(method_output_file)
		precisions.append(precision)
		recalls.append(recall)
		compression_ratios.append(compression_ratio)
		association_measures.append(association_measure)

	print('Noise level:', noise_level, ', number of files:', len(noise_level_files))
	# print('   Precision:', np.mean(precisions))
	# print('   Recall:', np.mean(recalls))
	# print('   F1:', 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)))
	print('   Compression ratio:', np.mean(compression_ratios))
	print('   Association measure:', np.mean(association_measures))



noise_levels = [0, 0.01, 0.03, 0.05]
for noise_level in noise_levels:
	eval_noise_level(noise_level)






