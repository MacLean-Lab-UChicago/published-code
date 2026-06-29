'''
generate null distributions of things to do with Functional Networks after resampling in time
loosely based on the method used in Krustal and MacLean (2013)
weighted random resampling method from Efraimidis and Spirakis (2006) - Algorithm A-Res
'''
import numpy as np
import matplotlib.pyplot as plt
import src.utils
from scipy.stats import pearsonr
import multiprocessing
from functools import partial
import os

def bootstrap_worker(boot_idx, n_neurons, n_trials, time_in_trial, activity, PETHs, PDF, FN_method, mouse_dir, all_spikes=None):
	'''
	(Internal subfunction) Performs one resampling of the neural data
	this is called in parallel by the main function
	'''
	if FN_method=='MI' and len(all_spikes)==0:
		raise Warning('need full Cascade distribution for Cascade binarization')

	# resample time for each neuron
	resampled = np.zeros((n_neurons, n_trials*time_in_trial))*np.nan
	for neuron in range(n_neurons):
		neuron_activity = activity[neuron, :]
		assert PDF.shape[0]==n_neurons, f'PDF tiled wrong; {PDF.shape}'
		PDF_cell = PDF[neuron, :]
		random_values = np.random.rand(n_trials*time_in_trial)
		keys = random_values**(1/PDF_cell)
		Cascade_sorted = np.sort(activity[neuron, :])
		resampled[neuron, np.argsort(keys)] = Cascade_sorted

	# calculate edges
	graph = np.zeros((n_neurons, n_neurons))*np.nan
	if FN_method=='pearson_corr':
		for x in range(n_neurons):
			for y in np.arange(x+1, n_neurons):
				corr = pearsonr(resampled[x, :], resampled[y, :], axis=None)
				graph[x,y] = corr.statistic
				graph[y,x] = corr.statistic
	
	elif FN_method=='MI':
		# binarize the Cascade inputs
		spikes = src.utils.binarize_spikes(all_spikes, resampled, percent_threshold=95)
		for x in range(n_neurons):
			for y in np.arange(x+1, n_neurons):
				mut_info = src.utils.calculate_MI(spikes[x, :], spikes[y, :])
				graph[x,y] = mut_info
				graph[y,x] = mut_info

	edge_weights = graph[np.triu_indices(n_neurons, 1)].flatten()
	return edge_weights

def bootstrap_edge_weights(n_neurons, trial_types, n_trials_dict, time_in_trial, activity_dict, PETHs, PDF, FN_method, mouse_dir, all_spikes=None, n_boostrap_samples=1000):
	'''
	calculates the edge weights using FN_method, parallelized over bootstrap samples

	args:
		n_neurons
		trial_types (to index into the dictionaries)
		n_trials_dict: dictionary of number of trials in each trial type
		time_in_trial
		activity_dict: dictionary of flattened Cascade activity across all trials for a given trial type
		PDF: probability density function (derived from PETH)
		FN_method: either pearson_corr or MI
		all_spikes: required for binarization for calculating MI
		n_bootstrap samples (int): number of resampling repeats

	returns:
		resampled_edge_weights: dictionary containing (BxP) edge weight matrices for each trial type
		average_weights: dictionary containing (B,) average edge weight for each trial type
	'''

	# Outer loop over trial types in serial
	resampled_edge_weights = {type:[] for type in trial_types}
	average_weights = {type:[] for type in trial_types}
	for type in trial_types:
		print(f'Processing {type} trials...')

		# Use partial to "freeze" arguments for the worker function
		worker_func = partial(
			bootstrap_worker,
			n_neurons=n_neurons, 
			n_trials=n_trials_dict[type], 
			time_in_trial=time_in_trial, 
			activity=activity_dict[type], 
			PETHs=PETHs,
			PDF=np.tile(PDF, n_trials_dict[type]), 
			FN_method=FN_method, 
			mouse_dir=mouse_dir,
			all_spikes=all_spikes
		)

		# Create a pool and map the work across CPU cores
		with multiprocessing.Pool() as pool:
			# map distributes the B bootstrap samples to the worker function
			# results is a list of B arrays of edge weights, each of length P (number of possible edges)
			results = pool.map(worker_func, range(n_boostrap_samples))

		# consolidate results, shape: (B, P)
		edge_weight_matrix = np.array(results)
		resampled_edge_weights[type] = edge_weight_matrix

		# find the mean edge weight for each boostrap
		if 'corr' in FN_method:
			# if a correlation, first do a Fisher Z transform, take the average, then transform back
			avg_weights = np.tanh(np.mean(np.arctanh(edge_weight_matrix), axis=1))
		else:
			# otherwise just take the average
			avg_weights = np.mean(edge_weight_matrix, axis=1)

		average_weights[type] = avg_weights

	return resampled_edge_weights, average_weights


mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
weight=10000 # controls how much the null model recapitulates PETHs (higher number means closer to the real PETH)

t_pre = 4 #frames (30 fps) 
t_post = 4 #frames
time_in_trial = t_pre+t_post

FN_method = 'pearson_corr' # pearson_corr or MI

for mouseID in mice:
	print(mouseID)
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	days = src.IO.get_carry_days(mouseID)

	if not os.path.isdir(mouse_dir + 'carry_analysis/figures/null/'):
		os.mkdir(mouse_dir + 'carry_analysis/figures/null/')

	reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, days)
	
	active_spks = [[] for i in range(len(days))]
	empty_spks = [[] for i in range(len(days))]

	if FN_method == 'MI':
		all_spks = [[] for i in range(len(days))]

	for i, day in enumerate(days):
		print(day)
		# load in the Cascade spikes
		calcium_data_path = mouse_dir + day
		if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
			s2p_fld = calcium_data_path + '/'
		else:
			s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'

		spks = np.load(s2p_fld + 'cascade_spks.npy')
		# index by the registered cells
		reg_spks = spks[reg_inds[i], :]
		n_neurons = reg_spks.shape[0]

		if FN_method=='MI':
			all_spks[i] = reg_spks

		# load in the behavioral times
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		s_carry_times = carry_times[carry_labels==1]
		empty_carry_times = carry_times[carry_labels==2]

		# grab relevant neural activity
		temp_a_spks = np.zeros((n_neurons, len(s_carry_times), time_in_trial))
		temp_e_spks = np.zeros((n_neurons, len(empty_carry_times), time_in_trial))
		for carry_i, carry in enumerate(s_carry_times.astype(int)):
			temp_a_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		for carry_i, carry in enumerate(empty_carry_times.astype(int)):
			temp_e_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		active_spks[i]=temp_a_spks
		empty_spks[i]=temp_e_spks

	# make NeuronsxTrialsxTime array of neural activity
	active_spikes = np.concatenate(active_spks, axis=1)
	empty_spikes = np.concatenate(empty_spks, axis=1)

	n_active_trials = active_spikes.shape[1]
	n_empty_trials = empty_spikes.shape[1]
	print('empty trials: ', n_empty_trials)
	print('active trials: ', n_active_trials)

	# remove neurons with only NaN for activity
	active_spikes_nonan = active_spikes[~np.isnan(active_spikes).any(axis=(1,2)), :]
	empty_spikes_nonan = empty_spikes[~np.isnan(active_spikes).any(axis=(1,2)), :]
	print(np.sum(np.isnan(active_spikes).any(axis=(1,2))), ' NaN-containing cells removed')
	n_neurons = np.sum(~np.isnan(active_spikes).any(axis=(1,2)))

	# calculate PETH per cell over both active and empty trials
	all_trials = np.concatenate((active_spikes_nonan, empty_spikes_nonan), axis=1)
	n_trials = all_trials.shape[1]
	print('n trials total: ', n_trials)
	PETHs = np.array(np.nanmean(all_trials, axis=1)) # should now be cellsxtime in trial
	assert PETHs.shape[0]==n_neurons and PETHs.shape[1]==time_in_trial, f'{PETHs.shape}'


	# convert the PETH into a probability density function of activity for each cell
	sum_all_activity = np.sum(PETHs, axis=1).reshape((n_neurons,1))
	assert sum_all_activity.shape[0]==n_neurons, f'{sum_all_activity.shape}'
	PDF = np.divide(PETHs, sum_all_activity)*weight

	# now, for each neuron and each trial type, 
	# flatten the Cascade spikes to get an array with length n_{type}_trials*time_in_trial
	active_activity = active_spikes_nonan.reshape((n_neurons, n_active_trials*time_in_trial))
	empty_activity = empty_spikes_nonan.reshape((n_neurons, n_empty_trials*time_in_trial))

	trial_types = ['active', 'empty']
	n_trials_dictionary = {'active':n_active_trials, 'empty':n_empty_trials}
	activity_dictionary={'active':active_activity, 'empty':empty_activity}

	if FN_method=='MI':
		all_spikes = np.hstack(all_spks)
	else:
		all_spikes=None

	resampled_edge_weights, average_weights = bootstrap_edge_weights(
		n_neurons, trial_types, n_trials_dictionary, time_in_trial, activity_dictionary, 
		PETHs, PDF, FN_method, mouse_dir, all_spikes=all_spikes, n_boostrap_samples=1000)

	# save the full null distribution network
	src.IO.save_pickle(mouse_dir + f'carry_analysis/weighted_PDF_{weight}_shuffled_{FN_method}_edge_weights.pkl', resampled_edge_weights)
	# and average functional connection weight per pair of neurons in the null model
	src.IO.save_pickle(mouse_dir + f'carry_analysis/weighted_PDF_{weight}_shuffled_{FN_method}_averaged_edge_weights.pkl', average_weights)
