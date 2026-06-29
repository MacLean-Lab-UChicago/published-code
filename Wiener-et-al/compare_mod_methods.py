'''
compare average firing rate and instant firing rate SVM single cell decoders
classify cells are time-dependent decodable, increased firing rate decodable, decreased firing rate decodable, or not decodable
'''
import numpy as np
import itertools
import src.utils
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ('mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549')
methods = ['SVM', 'SVM_avgfr']
calculate_proportions = True

time_varying_only = 0
trial_avged = 0
total_modulated = 0
n_neurons = 0

if calculate_proportions:
	# initialize dictionaries for cell classifications
	mod_counts = {key:np.zeros(len(mice)) for key in ['Time-Dependent', 'Increased', 'Decreased', 'Not Decodable']}
	save_dir = data_dir + '/results/cross_mouse_results/'
	t_pre=4
	t_post=4
	time_in_trial=t_pre+t_post

for mouse_i, mouseID in enumerate(mice):
	print(mouseID)
	days = src.IO.get_carry_days(mouseID)
	s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])

	# grab the decodable cells
	mod_cells_by_method = [[] for method in methods]
	for i, method in enumerate(methods):
		mod_cells = np.load(s2p_fld + method + '_active_grasp_modulated_cells.npy')
		mod_cells_idx = np.where(mod_cells)[0]
		mod_cells_by_method[i]=mod_cells_idx
	n_neurons+=len(mod_cells)
	
	combinations = itertools.combinations(range(len(methods)), 2)
	for i, j in combinations:
		method_1 = methods[i]
		method_2 = methods[j]

		mod_1 = mod_cells_by_method[i]
		mod_2 = mod_cells_by_method[j]

		unique_1 = [mod_cell for mod_cell in mod_1 if mod_cell not in mod_2]
		unique_2 = [mod_cell for mod_cell in mod_2 if mod_cell not in mod_1]
		overlapping = [mod_cell for mod_cell in mod_1 if mod_cell in mod_2]
		combined = np.unique(np.append(mod_1, mod_2))
		time_varying_only +=len(unique_1) # if only classified in the instant firing rate model
		trial_avged+=len(unique_2)+len(overlapping) # if classified in the average firing rate model or both models
		total_modulated+=len(combined) # all decodable cells

		print(method_1, method_2)
		print(f'{method_1}:  {len(unique_1)} unique modified cells')
		print(f'{len([mod_cell for mod_cell in unique_1 if mod_cell in red_cell_idx])} red cells')
		print(f'{method_2}: {len(unique_2)} unique modified cells')
		print(f'{len([mod_cell for mod_cell in unique_2 if mod_cell in red_cell_idx])} red cells')
		print('overlapping: ', len(overlapping))
		print(f'{len([mod_cell for mod_cell in overlapping if mod_cell in red_cell_idx])} red cells')
		print('total combined: ', len(combined))
		print('')

		if method_1=='SVM' and method_2=='SVM_avgfr':
			np.save(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy', combined)
			np.save(s2p_fld + 'SVM_time_averaged_active_grasp_modulated_cell_indices.npy', np.array(np.append(overlapping, unique_2)))
			np.save(s2p_fld + 'SVM_time_varying_active_grasp_modulated_cell_indices.npy', np.array(unique_1))

	if calculate_proportions:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		days = src.IO.get_carry_days(mouseID)
		reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, days)

		active_spks = [[] for i in range(len(days))]
		empty_spks = [[] for i in range(len(days))]

		for i, day in enumerate(days):
			# load in the Cascade spikes
			calcium_data_path = mouse_dir + day
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
			spks = np.load(s2p_fld + 'cascade_spks.npy')
			# index by the registered cells
			reg_spks = spks[reg_inds[i], :]
			n_neurons=reg_spks.shape[0]

			# load in the behavioral times
			carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
			carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
			active_carry_times = carry_times[carry_labels==1]
			empty_carry_times = carry_times[carry_labels==2]

			# grab active and empty neural activity
			temp_a_spks = np.zeros((n_neurons, len(active_carry_times), time_in_trial))
			temp_e_spks = np.zeros((n_neurons, len(empty_carry_times), time_in_trial))
			for carry_i, carry in enumerate(active_carry_times.astype(int)):
				temp_a_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
			for carry_i, carry in enumerate(empty_carry_times.astype(int)):
				temp_e_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
			active_spks[i]=temp_a_spks
			empty_spks[i]=temp_e_spks

		# make array of n_neuronsxn_trialsxtime_in_trial neural activity
		active = np.concatenate(active_spks, axis=1)
		empty = np.concatenate(empty_spks, axis=1)

		# take the average over trials for each cell
		a_mean = np.mean(active, axis=1)
		e_mean = np.mean(empty, axis=1)

		# take the average over timepoints for each cell
		a_meanmean = np.mean(a_mean, axis=1)
		e_meanmean = np.mean(e_mean, axis=1)

		# remove cells with no recorded activity
		nan_cells = np.isnan(a_meanmean)
		a_meanmean = a_meanmean[~nan_cells]
		e_meanmean = e_meanmean[~nan_cells]
		
		# remove NaN cells from decodable cell indices
		mod_cells = np.zeros(a_mean.shape[0], dtype=bool)
		mod_cell_idx = combined
		mod_cells[mod_cell_idx] = True
		mod_cells = mod_cells[~nan_cells]

		non_time_mod_cells = np.zeros(a_mean.shape[0], dtype=bool)
		non_time_mod_cells[np.append(overlapping, unique_2)] = True
		non_time_mod_cells = non_time_mod_cells[~nan_cells]
		
		time_mod_cells = np.zeros(a_mean.shape[0], dtype=bool)
		time_mod_cells[unique_1] = True
		time_mod_cells = time_mod_cells[~nan_cells]

		up_mod = np.logical_and(non_time_mod_cells, (a_meanmean>e_meanmean)) # if average firing rate decodable and active firing rate is greater than empty
		down_mod = np.logical_and(non_time_mod_cells, (e_meanmean>a_meanmean)) # if average firing rate decodable and active firing rate is less than empty
		mod_time = time_mod_cells

		assert (np.sum(up_mod)+np.sum(down_mod)+np.sum(mod_time))==np.sum(mod_cells), f'{np.sum(up_mod)} {np.sum(down_mod)} {np.sum(mod_time)} {np.sum(mod_cells)}'

		mod_counts['Not Decodable'][mouse_i]=np.sum(~mod_cells)
		mod_counts['Increased'][mouse_i]=np.sum(up_mod)
		mod_counts['Decreased'][mouse_i]=np.sum(down_mod)
		mod_counts['Time-Dependent'][mouse_i]=np.sum(mod_time)

if calculate_proportions:
	src.IO.save_pickle(save_dir + 'mod_counts.pkl', mod_counts)
	print(mod_counts)

print('n time varying: ', time_varying_only)
print('n time averaged: ', trial_avged)
print('total: ', total_modulated, '/', n_neurons)