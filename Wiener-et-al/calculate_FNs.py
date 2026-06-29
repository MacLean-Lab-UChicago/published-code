'''
generate the functional networks for active grasp carries and empty carries
'''

from scipy.stats import pearsonr, spearmanr
import numpy as np
import src.utils
import src.IO
import os
from pathlib import Path
data_dir = Path.cwd() / 'data'

# code parameters
FN_method = 'pearson_corr' # pearson_corr, spearman_corr, or MI
thresh_percent = 95 # percentile to threshold at for binarizing spikes, only needed for MI

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']


empty_normalized_weights = [[] for mouse in mice]
active_normalized_weights = [[] for mouse in mice]
for mouse_i, mouseID in enumerate(mice):
	mouse_dir = data_dir + '/neural/' + mouseID + '/'

	days = src.IO.get_carry_days(mouseID)
	t_pre = 4 #frames (30 fps) 
	t_post = 4 #frames

	# load in the aligned cells across days
	reg_inds = src.utils.load_registered_cells(mouse_dir, days)

	active_spks = [[] for i in range(len(days))]
	empty_spks = [[] for i in range(len(days))]

	if FN_method == 'MI':
		all_spks = [[] for i in range(len(days))]

	for i, day in enumerate(days):
		print(day)
		# load in the Cascade spikes
		calcium_data_path = mouse_dir + day
		s2p_fld = calcium_data_path + '/'
		
		spks = np.load(s2p_fld + 'cascade_spks.npy')
		# index by the registered cells
		reg_spks = spks[reg_inds[i], :]

		if FN_method=='MI':
			all_spks[i] = reg_spks

		# load in the behavioral times
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		s_carry_times = carry_times[carry_labels==1]
		empty_carry_times = carry_times[carry_labels==2]

		# grab relevant neural activity
		time_in_trial = t_pre+t_post
		temp_s_spks = [[] for carry_i in range(len(s_carry_times))]
		for carry_i, carry in enumerate(s_carry_times.astype(int)):
			temp_s_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		active_spks[i] = np.hstack(temp_s_spks)
		temp_empty_spks = [[] for carry_i in range(len(empty_carry_times))]
		for carry_i, carry in enumerate(empty_carry_times.astype(int)):
			temp_empty_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		empty_spks[i] = np.hstack(temp_empty_spks)

	active_spikes = np.hstack(active_spks)
	empty_spikes = np.hstack(empty_spks)

	# remove neurons with only NaN for activity
	active_spikes_nonan = active_spikes[~np.isnan(active_spikes).any(axis=1), :]
	empty_spikes_nonan = empty_spikes[~np.isnan(active_spikes).any(axis=1), :]

	# save out a boolean of the NaN cells
	print(np.sum(np.isnan(active_spikes).any(axis=1)), ' NaN-containing cells removed')
	print(np.where(np.isnan(active_spikes).any(axis=1))[0])
	np.save(s2p_fld + 'NaN_containing_cells_bool.npy', np.isnan(active_spikes).any(axis=1))

	n_neurons = active_spikes_nonan.shape[0]
	print('n cells: ', n_neurons)
	active_graph = np.zeros((n_neurons, n_neurons))
	empty_graph = np.zeros((n_neurons, n_neurons))

	if FN_method=='pearson_corr':
		# note: this outputs a signed, undirected graph
		for x in range(n_neurons):
			for y in range(n_neurons):
				s_corr = pearsonr(active_spikes_nonan[x, :], active_spikes_nonan[y, :], axis=None)
				active_graph[x,y] = s_corr.statistic
				d_corr = pearsonr(empty_spikes_nonan[x, :], empty_spikes_nonan[y, :], axis=None)
				empty_graph[x,y] = d_corr.statistic

	elif FN_method=='spearman_corr':
		# note: this outputs a signed, undirected graph
		for x in range(n_neurons):
			for y in range(n_neurons):
				s_corr = spearmanr(active_spikes_nonan[x, :], active_spikes_nonan[y, :], axis=None)
				active_graph[x,y] = s_corr.statistic
				d_corr = spearmanr(empty_spikes_nonan[x, :], empty_spikes_nonan[y, :], axis=None)
				empty_graph[x,y] = d_corr.statistic

	elif FN_method=='MI':
		# note: this outputs an unsigned, undirected graph
		all_spikes = np.hstack(all_spks)
		# binarize the spikes from Cascade based on a percentile threshold
		active_spikes = src.utils.binarize_spikes(all_spikes, active_spikes_nonan, thresh_percent)
		empty_spikes = src.utils.binarize_spikes(all_spikes, empty_spikes_nonan, thresh_percent)
		for x in range(n_neurons):
			for y in range(n_neurons):
				active_graph[x,y] = src.utils.calculate_MI(active_spikes[x, :], active_spikes[y, :])
				empty_graph[x,y] = src.utils.calculate_MI(empty_spikes[x, :], empty_spikes[y, :])
	else:
		raise Warning('not a graph inference method that I wrote code for')

	np.save(s2p_fld + f'active_FN_{FN_method}.npy', active_graph)
	np.save(s2p_fld + f'empty_FN_{FN_method}.npy', empty_graph)
