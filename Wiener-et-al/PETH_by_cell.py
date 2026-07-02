'''
generates PETHs for individual cells during different carry classes
'''
import numpy as np
import matplotlib.pyplot as plt
import src.utils
import os
import scipy.stats as stats
import random
from pathlib import Path
data_dir = Path.cwd() / 'data'

neurons_to_plot = 'max' # how many neurons to plot or all of them or a list of indices
show_plots = False
plot_non_mod = False
plot_mod = True
full_population = False
all_mice_full_pop = True

multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'

t_pre = 9 #frames (30 fps) 
t_post = 8 #frames
time_in_trial = t_pre+t_post
time = np.linspace(-(t_pre-4)*1000/30, (t_post+4)*1000/30, time_in_trial)

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

all_mice_all_cells_active = np.zeros((0, 0, time_in_trial))*np.nan
all_mice_all_cells_empty = np.zeros((0, 0, time_in_trial))*np.nan

for mouseID in mice:
	drive = data_dir + '/neural'
	mouse_dir = drive + '/' + mouseID + '/'
	fig_savepath = mouse_dir + 'carry_analysis/figures/cell_PETHs/'

	if not os.path.isdir(fig_savepath):
		os.mkdir(fig_savepath)
	
	days = src.IO.get_carry_days(mouseID)

	reg_inds = src.utils.load_registered_cells(mouse_dir, days)

	active_spks = [[] for i in range(len(days))]
	empty_spks = [[] for i in range(len(days))]

	for i, day in enumerate(days):
		# load in the Cascade spikes
		calcium_data_path = mouse_dir + day
		s2p_fld = calcium_data_path + '/'
		
		spks = np.load(s2p_fld + 'cascade_spks.npy')
		# index by the registered cells
		reg_spks = spks[reg_inds[i], :]
		n_neurons=reg_spks.shape[0]

		# load in the behavioral times
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		active_carry_times = carry_times[carry_labels==1]
		empty_carry_times = carry_times[carry_labels==2]

		# grab relevent neural data
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

	# combine across mice
	active_with_NaNs = np.concatenate((active, np.zeros((all_mice_all_cells_active.shape[0], active.shape[1], active.shape[2]))*np.nan), axis=0)
	empty_with_NaNs = np.concatenate((empty, np.zeros((all_mice_all_cells_empty.shape[0], empty.shape[1], empty.shape[2]))*np.nan), axis=0)
	all_mice_all_cells_active_with_NaNs = np.concatenate((np.zeros((active.shape[0], all_mice_all_cells_active.shape[1], all_mice_all_cells_active.shape[2]))*np.nan, all_mice_all_cells_active), axis=0)
	all_mice_all_cells_empty_with_NaNs = np.concatenate((np.zeros((empty.shape[0], all_mice_all_cells_empty.shape[1], all_mice_all_cells_empty.shape[2]))*np.nan, all_mice_all_cells_empty), axis=0)

	all_mice_all_cells_active = np.concatenate([all_mice_all_cells_active_with_NaNs, active_with_NaNs], axis=1)
	all_mice_all_cells_empty = np.concatenate([all_mice_all_cells_empty_with_NaNs, empty_with_NaNs], axis=1)

	# load cell properties
	mod_cells_idx = np.load(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy')
	mod_cells = np.zeros(n_neurons, dtype=bool)
	mod_cells[mod_cells_idx] = True

	if full_population: # plot the population activity per mouse
		# remove cells with NaNs
		print(empty.shape)
		non_nan_cells = ~np.isnan(active).any(axis=2).any(axis=1)
		print(np.sum(non_nan_cells))
		print(active_pop_fr.shape)
		empty_pop_fr = np.mean(empty[non_nan_cells, :, :], axis=0)

		# find the trial-averaged population mean and SEM
		a_mean = np.mean(active_pop_fr, axis=0)*30
		a_sem = stats.sem(active_pop_fr, axis=0)*30
		e_mean = np.mean(empty_pop_fr, axis=0)*30
		e_sem = stats.sem(empty_pop_fr, axis=0)*30

		# plot 
		plt.plot(time, a_mean, color='indianred', label='active grasp')
		plt.fill_between(time, a_mean-a_sem, a_mean+a_sem, color='indianred', alpha=0.4)
		plt.plot(time, e_mean, color='k', label='empty grasp')
		plt.fill_between(time, e_mean-e_sem, e_mean+e_sem, color='k', alpha=0.4)
		plt.axvline(0, ls='--', color='k')
		plt.axvline(8*1000/30, ls='--', color='k')
		plt.xlabel('Time from Carry Start (ms)')
		plt.title('Population-Averaged PETH')
		plt.ylabel('Firing Rate (Hz)')
		plt.legend(loc='upper right')
		plt.savefig(fig_savepath + 'population_averaged_PETH.png')
		plt.show()

		# find the average firing rates per trial (in individual neurons)
		active_avg_frs = np.mean(active[non_nan_cells, :, :], axis=2).flatten()*30
		empty_avg_frs = np.mean(empty[non_nan_cells, :, :], axis=2).flatten()*30

		# plot a histogram 
		bins=20
		bin_range=(0, np.max(np.append(active_avg_frs, empty_avg_frs)))
		plt.hist(active_avg_frs, bins=bins, range=bin_range, color='indianred', alpha=0.5, label='active grasp')
		plt.hist(empty_avg_frs, bins=bins, range=bin_range, color='k', alpha=0.5, label='empty grasp')
		plt.axvline(np.mean(active_avg_frs), color='indianred')
		plt.axvline(np.mean(empty_avg_frs), color='k')
		plt.xlabel('Firing Rate (Hz)')
		plt.ylabel('Count')
		ks = stats.kstest(active_avg_frs, empty_avg_frs, nan_policy='omit')
		plt.title(f'd={(np.mean(active_avg_frs)-np.mean(empty_avg_frs)):4f}, p={ks.pvalue:.4f}')
		plt.suptitle('Distribution of Average Firing Rates')
		plt.legend()
		plt.savefig(fig_savepath + 'Trial Firing Rates By Condition.png')
		plt.show()

		# plot a histogram with a log scale
		plt.hist(active_avg_frs, bins=bins, range=bin_range, color='indianred', alpha=0.5, label='active grasp')
		plt.hist(empty_avg_frs, bins=bins, range=bin_range, color='k', alpha=0.5, label='empty grasp')
		plt.axvline(np.mean(active_avg_frs), color='indianred')
		plt.axvline(np.mean(empty_avg_frs), color='k')
		plt.xlabel('Firing Rate (Hz)')
		plt.ylabel('Count')
		ks = stats.kstest(active_avg_frs, empty_avg_frs, nan_policy='omit')
		plt.title(f'd={(np.mean(active_avg_frs)-np.mean(empty_avg_frs)):4f}, p={ks.pvalue:.4f}')
		plt.suptitle('Distribution of Average Firing Rates')
		plt.legend()
		plt.yscale('log')
		plt.savefig(fig_savepath + 'Trial Firing Rates By Condition y log scale.png')
		plt.show()

		# plot individual neuron PETHs
		if plot_mod:
			if neurons_to_plot=='max':
				cells = mod_cells_idx
			elif type(neurons_to_plot)==list:
				cells = np.array(neurons_to_plot, dtype=int)
			else:	
				cells_idx=np.array(random.sample(range(len(mod_cells_idx)), k=neurons_to_plot)).astype(int)
				cells = mod_cells_idx[cells_idx]
			for cell in cells:
				# find single cell trial-averaged activity and SEM
				a_mean = np.mean(active[cell, :, :], axis=0)*30
				a_sem = stats.sem(active[cell, :, :], axis=0)*30
				e_mean = np.mean(empty[cell, :, :], axis=0)*30
				e_sem = stats.sem(active[cell, :, :], axis=0)*30

				# plot
				fig, ax = plt.subplots()
				ax.plot(time, a_mean, color='indianred', label='active grasp')
				ax.fill_between(time, a_mean-a_sem, a_mean+a_sem, color='indianred', alpha=0.4)
				ax.plot(time, e_mean, color='k', label='empty grasp')
				ax.fill_between(time, e_mean-e_sem, e_mean+e_sem, color='k', alpha=0.4)
				ax.set_ylabel('Firing Rate (Hz)')
				ax.axvline(0, ls='--', color='k')
				ax.axvline(8*1000/30, ls='--', color='k')
				ax.set_xlabel('Time from Carry Start (ms)')
				ax.set_xlim((time[0], time[-1]))
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				plt.savefig(fig_savepath + f'mod_cells/Cell {cell} PETH.pdf', dpi=660)
				if show_plots:
					plt.show()
				else:
					plt.close()

		if plot_non_mod:
			non_mod_idx = np.where(~mod_cells)[0]
			if neurons_to_plot=='max':
				cells=non_mod_idx
			else:
				cells_idx=np.array(random.sample(range(len(mod_cells_idx)), k=neurons_to_plot)).astype(int)
				cells = non_mod_idx[cells_idx]
			for cell in cells:
				# find single cell trial-averaged activity and SEM
				a_mean = np.mean(active[cell, :, :], axis=0)*30
				a_sem = stats.sem(active[cell, :, :], axis=0)*30
				e_mean = np.mean(empty[cell, :, :], axis=0)*30
				e_sem = stats.sem(active[cell, :, :], axis=0)*30

				# plot
				plt.plot(time, a_mean, color='indianred', label='active grasp')
				plt.fill_between(time, a_mean-a_sem, a_mean+a_sem, color='indianred', alpha=0.4)
				plt.plot(time, e_mean, color='k', label='empty grasp')
				plt.fill_between(time, e_mean-e_sem, e_mean+e_sem, color='k', alpha=0.4)
				plt.legend(loc='upper right')
				plt.ylabel('Firing Rate (Hz)')
				plt.axvline(0, ls='--', color='k')
				plt.axvline(8*1000/30, ls='--', color='k')
				plt.xlabel('Time from Carry Start (ms)')
				plt.savefig(fig_savepath + f'non_mod_cells/Cell {cell} PETH.png')
				if show_plots:
					plt.show()
				else:
					plt.clf()

if all_mice_full_pop:
	print(all_mice_all_cells_active.shape)
	non_nan_cells = ~np.isnan(all_mice_all_cells_active).all(axis=2).all(axis=1)
	print(np.sum(non_nan_cells))

	# find the average population activity across all cells (all mice)
	active_pop_fr = np.nanmean(all_mice_all_cells_active[non_nan_cells, :, :], axis=0)
	print(active_pop_fr.shape)
	empty_pop_fr = np.nanmean(all_mice_all_cells_empty[non_nan_cells, :, :], axis=0)
	
	# trial-average the population activity and find the SEM
	a_mean = np.mean(active_pop_fr, axis=0)*30
	a_sem = stats.sem(active_pop_fr, axis=0)*30
	e_mean = np.mean(empty_pop_fr, axis=0)*30
	e_sem = stats.sem(empty_pop_fr, axis=0)*30

	# plot the population-averaged PETH
	fig, ax = plt.subplots()
	ax.plot(time, a_mean, color='indianred', label='active grasp')
	ax.fill_between(time, a_mean-a_sem, a_mean+a_sem, color='indianred', alpha=0.4)
	ax.plot(time, e_mean, color='k', label='empty grasp')
	ax.fill_between(time, e_mean-e_sem, e_mean+e_sem, color='k', alpha=0.4)
	ax.axvline(0, ls='--', color='k')
	ax.axvline(8*1000/30, ls='--', color='k')
	ax.set_xlabel('Time from Carry Start (ms)')
	#ax.set_title('Population-Averaged PETH')
	ax.set_ylabel('Firing Rate (Hz)')
	ax.set_xlim((time[0], time[-1]))
	#plt.legend(loc='upper right')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig(multimouse_fig_dir + 'population_averaged_PETH.pdf', dpi=660)
	plt.show()