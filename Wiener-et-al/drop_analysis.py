'''
look at drop PETHs, both full population and individual cells
do ANOVA for pre- vs post-drop to find drop-modulated cells
	the rationale here is that since drops happen at many points throughout the carry,
	we can "average out" the kinematic component 
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import src.utils
import os
import seaborn as sns

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

do_calcs=True
plot_cell_PETHs = True
plot_population_activity = True
plot_population_PETH = True
do_ANOVA = True
encoding_model=True
examine_mod_cells = True
show_figs=True
verbose=True

t_pre = 5 # frames (30 fps)
t_post = 10 # frames (30 fps)
time_in_trial = t_pre+t_post

splitter='Shuffle'
if splitter=='Shuffle':
	n_folds=10
	test_size=0.3
elif splitter=='KFold':
	n_repeats=10
	n_folds=2

multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/drops/'

if do_calcs:
	for mouse_i, mouseID in enumerate(mice):
		carry_days = src.IO.get_carry_days(mouseID)
		drop_spks = []
		active_average_spks = []
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/drops/'
		reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, carry_days)
		tot_trials = 0
		if not os.path.isdir(save_dir):
				os.mkdir(save_dir)
				os.mkdir(save_dir + 'figures/')
				os.mkdir(save_dir + 'figures/cell_PETHs/')

		for day_i, day in enumerate(carry_days):
			s2p_fld = src.IO.get_s2p_fld(mouseID, day)
			spks = np.load(s2p_fld + 'cascade_spks.npy')
			# index only the cells that exist across all days
			spks = spks[reg_inds[day_i], :]
			#print(spks.shape[0])
			# n_neurons = spks.shape[0]
			# print(n_neurons)
			try:
				drop_times = np.load(s2p_fld + 'calcium_drop_times.npy')
			except:
				print(f'no drops for {mouseID} on {day}')
				continue
			
			n_trials=len(drop_times)
			tot_trials+=n_trials
			n_neurons = spks.shape[0]

			# grab drop-specific neural data
			drop_spikes = np.zeros((n_neurons, time_in_trial, n_trials))
			for drop_i, drop in enumerate(drop_times.astype(int)):
				drop_spikes[:, :, drop_i] = spks[:, (drop-t_pre):(drop+t_post)]
			drop_spks.append(drop_spikes)

		drop_spks = np.concatenate(drop_spks, axis=2)
		
		# remove cells with NaNs
		NaN_cells = np.load(s2p_fld + 'NaN_containing_cells_bool.npy')
		red_cells_bool = red_cells[~NaN_cells]
		red_cells = src.utils.adjust_indices_for_NaN_cell_removal(np.where(red_cells)[0], np.where(NaN_cells)[0])
		drop_spks = drop_spks[~NaN_cells, :, :] # remove cells with NaNs
		n_neurons = drop_spks.shape[0]
		print('after dropping NaNs, ', n_neurons, ' neurons')

		trial_average = np.nanmean(drop_spks, axis=2)*30
		std_error = stats.sem(drop_spks, axis=2, nan_policy='omit')*30
		time = time = np.linspace(-t_pre*1000/30, t_post*1000/30, time_in_trial)
		print('n trials: ', tot_trials)

		if plot_cell_PETHs:
			# plot each neuron's PETH (mean+-SEM)
			for neuron in range(n_neurons):
				plt.plot(time, trial_average[neuron, :], color='b')
				plt.fill_between(time, trial_average[neuron, :]-std_error[neuron, :], trial_average[neuron, :]+std_error[neuron, :], color='b', alpha=0.4)
				plt.axvline(0, ls='--', color='k')
				plt.xlabel('Time From Drop (ms)')
				plt.ylabel('Firing Rate (Hz)')
				if neuron in red_cells:
					plt.title(f'INH Cell {neuron}')
				else:
					plt.title(f'EXC Cell {neuron}')
				plt.savefig(f'{save_dir}figures/cell_PETHs/Cell {neuron} All Days Drop PETH.png')
				plt.close()


		if plot_population_activity:
			# plot each population average PETH (mean+-SEM)
			pop_avg = np.nanmean(trial_average, axis=0)
			pop_error = stats.sem(np.nanmean(drop_spks*30, axis=0), axis=1)
			plt.plot(time, pop_avg, color='b')
			plt.fill_between(time, pop_avg+pop_error, pop_avg-pop_error, color='b', alpha=0.4)
			plt.axvline(0, ls='--', color='k')
			plt.xlabel('Time From Drop (ms)')
			plt.ylabel('Firing Rate (Hz)')
			plt.title('Population Activity in Response to a Drop')
			plt.savefig(f'{save_dir}figures/All Days Population PETH.png')
			if show_figs:
				plt.show()
			else:
				plt.close()

		if plot_population_PETH:
			# plot tiled PETH (each row is a single cell's PETH)
			F_sorted, indices = src.utils.plot_PETH(spks,drop_times,t_pre=t_pre,t_post=t_post,sort=True,sort_ind=None,norm=True)
			plt.imshow(F_sorted, aspect='auto')
			locations = [0-.5, 0+t_pre, t_pre+t_post-1]
			labels = [-(t_pre-.5)*1000/30, 0, (t_post-1)*1000/30]
			plt.xticks(locations, labels=labels)
			plt.xlabel('Time (ms)')
			plt.ylabel('Neurons')
			plt.title('PETH of Drop')
			plt.colorbar()
			plt.savefig(f'{save_dir}figures/All Days Drop PETH.png')
			if show_figs:
				plt.show()
			else:
				plt.close()

			F_sorted, indices = src.utils.plot_PETH(spks,drop_times,t_pre=t_pre,t_post=t_post,sort=True,sort_ind=None,norm=False)
			plt.imshow(F_sorted, aspect='auto')
			plt.xticks(locations, labels=labels)
			plt.xlabel('Time (ms)')
			plt.ylabel('Neurons')
			plt.title('PETH of Drop')
			plt.colorbar()
			plt.savefig(f'{save_dir}figures/All Days Drop PETH non-normalized.png')
			if show_figs:
				plt.show()
			else:
				plt.close()

		if do_ANOVA:
			# do ANOVA to find drop-modulated cells
			pre_drop = drop_spks[:, 0:5, :].reshape((n_neurons, 5*tot_trials)) # 5 frames before the drop
			post_drop = drop_spks[:, 5:10, :].reshape((n_neurons, 5*tot_trials)) # 5 frames after the drop
			anova = stats.f_oneway(pre_drop, post_drop, axis=1, nan_policy='omit')
			pval = anova.pvalue
			#print(pval)
			bonferroni_mod_cells = pval<(.05/n_neurons)
			print(np.where(bonferroni_mod_cells))
			print('n mod cells (w/ Bonferroni correction): ', np.sum(bonferroni_mod_cells))
			p_adjusted = stats.false_discovery_control(pval[~np.isnan(pval)], method='bh')
			bh_mod_cells = p_adjusted<.05
			print(np.where(bh_mod_cells))
			print('n mod cells (w/ BH False Discovery Rate correction): ', np.sum(bh_mod_cells))
			np.save(f'{save_dir}{mouseID}_all_days_drop_bon_mod_cells.npy', bonferroni_mod_cells)
			np.save(f'{save_dir}{mouseID}_all_days_drop_BH_mod_cells.npy', bh_mod_cells)


if examine_mod_cells:
	for mouseID in mice:
		print(mouseID)

		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])
		
		# load in cell classifications 
		NaN_cells = np.load(s2p_fld + 'NaN_containing_cells_bool.npy')
		NaN_cells_idx = np.where(NaN_cells)[0]
		mod_cells = np.load(f'{mouse_dir}carry_analysis/drops/{mouseID}_all_days_drop_BH_mod_cells.npy')
		mod_cells_idx = np.where(mod_cells)[0]
		if len(mod_cells_idx)==len(NaN_cells):
			mod_cells_idx = src.IO.adjust_indices_for_NaN_cell_removal(mod_cells_idx, NaN_cells_idx)
		red_cells = src.utils.load_red_cells(mouse_dir, days)
		red_cells_idx = np.where(red_cells)[0]
		red_cells_idx = src.utils.adjust_indices_for_NaN_cell_removal_with_mouseID(red_cells_idx, mouseID)
		decodable_cells = src.utils.get_informative_cells_no_NaNs(s2p_fld)
		#print('NaNs: ', NaN_cells_idx)
		
		# print out the number of drop-mod cells per subtype category (inhibitory and active-empty decodable)
		print('mod cells: ', len(mod_cells_idx))
		print('mod indices: ', mod_cells_idx)
		print('inhibitory mod cells: ', len([i for i in mod_cells_idx if i in red_cells_idx]))
		print('decodable mod cells: ', len([i for i in mod_cells_idx if i in decodable_cells]))
		print('')


# plot drop-applied encoding results
if encoding_model:
	multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
	which_measure = 'median'
	print(which_measure)
	
	r2_dict={'mouse':[], 'r2':[], 'pre_post':[], 'category':[]}
	all_folds_dict={'mouse':[], 'r2':[], 'pre_post':[], 'category':[]}
	for mouseID in mice:
		print('')
		print(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		s2p_fld = src.IO.get_s2p_fld(mouseID, src.IO.get_carry_days(mouseID)[-1])
		
		# grab data and arrange into dict by pre- or post-drop and active or empty training
		for category in ['active', 'empty']:
			predrop_r2 = np.load(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_predrop_r2_values.npy')
			postdrop_r2 = np.load(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_postdrop_r2_values.npy')
			all_folds_dict['r2'].extend(predrop_r2.flatten())
			all_folds_dict['mouse'].extend(np.ones(len(predrop_r2.flatten()), dtype=object)*mouseID)
			all_folds_dict['pre_post'].extend(np.ones(len(predrop_r2.flatten()), dtype=object)*'pre-drop')
			all_folds_dict['category'].extend(np.ones(len(predrop_r2.flatten()), dtype=object)*category)
			# print(np.min(r2_values), np.max(r2_values))
			if which_measure=='median':
				med_r2 = np.median(predrop_r2, axis=1) # median over CV folds
			elif which_measure=='mean':
				med_r2 = np.mean(predrop_r2, axis=1) # mean over CV folds
			# print(np.median(med_r2))
			# print(np.min(avg_r2), np.max(avg_r2))
			r2_dict['r2'].extend(med_r2)
			r2_dict['mouse'].extend(np.ones(len(med_r2), dtype=object)*mouseID)
			r2_dict['pre_post'].extend(np.ones(len(med_r2), dtype=object)*'pre-drop')
			r2_dict['category'].extend(np.ones(len(med_r2), dtype=object)*category)

			all_folds_dict['r2'].extend(postdrop_r2.flatten())
			all_folds_dict['mouse'].extend(np.ones(len(postdrop_r2.flatten()), dtype=object)*mouseID)
			all_folds_dict['pre_post'].extend(np.ones(len(postdrop_r2.flatten()), dtype=object)*'post-drop')
			all_folds_dict['category'].extend(np.ones(len(postdrop_r2.flatten()), dtype=object)*category)
			if which_measure=='median':
				post_med_r2 = np.median(postdrop_r2, axis=1) # median over CV folds
			elif which_measure=='mean':
				post_med_r2 = np.mean(postdrop_r2, axis=1) # mean over CV folds
			r2_dict['r2'].extend(post_med_r2)
			r2_dict['mouse'].extend(np.ones(len(post_med_r2), dtype=object)*mouseID)
			r2_dict['pre_post'].extend(np.ones(len(post_med_r2), dtype=object)*'post-drop')
			r2_dict['category'].extend(np.ones(len(post_med_r2), dtype=object)*category)

			
			# find cells better modelled pre- or post-drop
			better_pre = [i for i in range(len(med_r2)) if med_r2[i]>0 and med_r2[i]>post_med_r2[i]]
			better_post = [i for i in range(len(med_r2)) if post_med_r2[i]>0 and post_med_r2[i]>med_r2[i]]

			print(category)
			print(f'{len(better_pre)} modelled better pre, {len(better_post)} modelled better post, of {len(med_r2)} cells')

		if verbose:
			# find cells better for each sub-category (empty+pre-drop, empty+post-drop, active+pre-drop, active+post-drop)
			better_empty_pre = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_empty_post = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_active_pre = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			better_active_post = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]

			print(f'for pre-drop: {len(better_empty_pre)} better empty, {len(better_active_pre)} better active')
			print(f'for post-drop: {len(better_empty_post)} better empty, {len(better_active_post)} better active')

			# find cells better for active or empty (overall, both in pre- and post-drop testing)
			better_empty = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_active = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			
			print(f'overall, both pre and post: {len(better_empty)} better empty, {len(better_active)} better active')

			# find cells better modeled active pre-drop and empty post-drop
			better_empty_pre_active_post = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			better_active_pre_empty_post = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]

			print(f'better active pre, empty post: {len(better_active_pre_empty_post)}, better empty pre, active post {len(better_empty_pre_active_post)}')

			# find how many cells are modeled above chance in at least one of the four categories
			cell_modeled_at_some_point = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				or np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				or np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				or np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0]
			
			print(f'{len(cell_modeled_at_some_point)}/{len(med_r2)} cells are well-modeled in at least one condition in the matrix of [pre/post] and [active/empty]')

			# find how many cells are modeled above chance for each category
			cell_modeled_pre_empty = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0]
			cell_modeled_post_empty = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0]
			cell_modeled_pre_active = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0]
			cell_modeled_post_active = [i for i in range(len(med_r2)) if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0]

			print(f'pre, empty:{len(cell_modeled_pre_empty)}; post, empty:{len(cell_modeled_post_empty)}')
			print(f'pre, active:{len(cell_modeled_pre_active)}; post, active:{len(cell_modeled_post_active)}')

			# see if drop-modulated cells are more/less well modelled compared to other cells
			mod_cells = np.load(f'{mouse_dir}carry_analysis/drops/{mouseID}_all_days_drop_BH_mod_cells.npy')
			NaN_cells = np.load(s2p_fld + 'NaN_containing_cells_bool.npy')
			NaN_cells_idx = np.where(NaN_cells)[0]
			mod_cells_idx = np.where(mod_cells)[0]
			if len(mod_cells)==len(NaN_cells):
				mod_cells_idx = src.utils.adjust_indices_for_NaN_cell_removal(mod_cells_idx, NaN_cells_idx)

			better_empty_pre = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_empty_post = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_active_pre = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			better_active_post = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]

			print(f'MOD CELLS, for pre-drop: {len(better_empty_pre)} better empty, {len(better_active_pre)} better active')
			print(f'MOD CELLS, for post-drop: {len(better_empty_post)} better empty, {len(better_active_post)} better active')

			better_empty = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]
			better_active = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			
			print(f'MOD CELLS, overall, both pre and post: {len(better_empty)} better empty, {len(better_active)} better active')

			better_empty_pre_active_post = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0 
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]]
			better_active_pre_empty_post = [i for i in mod_cells_idx if 
				np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>0
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]
				and np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][i]>np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][i]]

			print(f'MOD CELLS, better active pre, empty post: {len(better_active_pre_empty_post)}, better empty pre, active post {len(better_empty_pre_active_post)}')

			well_modelled_mod = [i for i in mod_cells_idx if i in cell_modeled_at_some_point]

			print(f'MOD CELLS {len(well_modelled_mod)}/{len(mod_cells_idx)} cells are well-modeled in at least one condition in the matrix of [pre/post] and [active/empty]')

			mod_cell_modeled_pre_empty = [i for i in mod_cells_idx if i in cell_modeled_pre_empty]
			mod_cell_modeled_post_empty = [i for i in mod_cells_idx if i in cell_modeled_post_empty]
			mod_cell_modeled_pre_active = [i for i in mod_cells_idx if i in cell_modeled_pre_active]
			mod_cell_modeled_post_active = [i for i in mod_cells_idx if i in cell_modeled_post_active]

			print(f'pre, empty:{len(mod_cell_modeled_pre_empty)}; post, empty:{len(mod_cell_modeled_post_empty)}')
			print(f'pre, active:{len(mod_cell_modeled_pre_active)}; post, active:{len(mod_cell_modeled_post_active)}')


	print(np.max(r2_dict['r2']))
	print(len(r2_dict['r2']))

	#violin plot of pre vs post drop modeling by mouse
	sns.violinplot(data=r2_dict, x='mouse', y='r2', hue='pre_post', split=True, inner='quart', gap=0.06, cut=0)
	#sns.swarmplot(data=r2_dict, x='mouse', y='r2', hue='couplings', alpha=0.4)
	plt.ylim(-.05, 1)
	plt.title('Ridge Linear Regression Performance on Drops with Couplings By Mouse')
	if show_figs:
		plt.show()
	else:
		plt.close()

	print(np.max(r2_dict['r2']))
	print(len(r2_dict['r2']))
	# violin plot of active vs empty trained modeling by mouse
	sns.violinplot(data=r2_dict, x='mouse', y='r2', hue='category', split=True, inner='quart', gap=0.06, cut=0)
	#sns.swarmplot(data=r2_dict, x='mouse', y='r2', hue='couplings', alpha=0.4)
	plt.ylim(-.05, 1)
	plt.title('Ridge Linear Regression Performance on Drops with Couplings By Mouse')
	if show_figs:
		plt.show()
	else:
		plt.close()

	# same but for all folds (rather than median across folds)
	print(np.max(all_folds_dict['r2']))
	print(len(all_folds_dict['r2']))
	sns.violinplot(data=all_folds_dict, x='mouse', y='r2', hue='pre_post', split=True, inner='quart', gap=0.06, cut=0)
	#sns.swarmplot(data=r2_dict, x='mouse', y='r2', hue='couplings', alpha=0.4)
	plt.ylim(-.05, 1)
	plt.title('Ridge Linear Regression Performance on Drops with Couplings By Mouse')
	if show_figs:
		plt.show()
	else:
		plt.close()

	print(np.max(all_folds_dict['r2']))
	print(len(all_folds_dict['r2']))
	sns.violinplot(data=all_folds_dict, x='mouse', y='r2', hue='category', split=True, inner='quart', gap=0.06, cut=0)
	#sns.swarmplot(data=r2_dict, x='mouse', y='r2', hue='couplings', alpha=0.4)
	plt.ylim(-.05, 1)
	plt.title('Ridge Linear Regression Performance on Drops with Couplings By Mouse')
	if show_figs:
		plt.show()
	else:
		plt.close()

	# combined across mice
	# active vs empty trained models, cv r^2
	sns.violinplot(data=r2_dict, x='category', y='r2', cut=0)
	result = stats.ttest_ind(np.array(r2_dict['r2'])[np.array(r2_dict['category'], dtype=object)=='active'], np.array(r2_dict['r2'])[np.array(r2_dict['category'], dtype=object)=='empty'])
	plt.ylim(-.05, 1)
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_figs:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')

	# combined across mice
	# active vs empty trained models, all folds
	sns.violinplot(data=all_folds_dict, x='category', y='r2', cut=0)
	result = stats.ttest_ind(np.array(all_folds_dict['r2'])[np.array(all_folds_dict['category'], dtype=object)=='active'], np.array(all_folds_dict['r2'])[np.array(all_folds_dict['category'], dtype=object)=='empty'])
	plt.ylim(-.05, 1)
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_figs:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')

	# combined across mice
	# pre- vs post-drop applied model performance
	sns.violinplot(data=r2_dict, x='pre_post', y='r2', hue='category', cut=0)
	result = stats.ttest_ind(np.array(r2_dict['r2'])[np.array(r2_dict['pre_post'], dtype=object)=='pre-drop'], np.array(r2_dict['r2'])[np.array(r2_dict['pre_post'], dtype=object)=='post-drop'])
	plt.legend()
	plt.ylim(-.05, 1)
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_figs:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')

	# combined across mice, active vs empty training and pre- vs post-drop applications
	sns.violinplot(data=r2_dict, x='category', y='r2', hue='pre_post', cut=0)
	result = stats.ttest_ind(np.array(r2_dict['r2'])[np.array(r2_dict['category'], dtype=object)=='active'], np.array(r2_dict['r2'])[np.array(r2_dict['category'], dtype=object)=='empty'])
	plt.legend()
	plt.ylim(-.05, 1)
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_figs:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')

	# combined across mice
	# pre- vs post-drop applied model performance, all folds
	sns.violinplot(data=all_folds_dict, x='pre_post', y='r2', cut=0)
	result = stats.ttest_ind(np.array(all_folds_dict['r2'])[np.array(all_folds_dict['pre_post'], dtype=object)=='pre-drop'], np.array(all_folds_dict['r2'])[np.array(all_folds_dict['pre_post'], dtype=object)=='post-drop'])
	plt.ylim(-.05, 1)
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_figs:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')

	mod_cells_by_mouse = {mouseID:[] for mouseID in mice}
	# scatterplot per mouse
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/drops/'
		fig_save_dir = save_dir+'figures/'
		mod_cells = np.load(f'{save_dir}{mouseID}_all_days_drop_BH_mod_cells.npy')
		mod_cells = src.utils.adjust_indices_for_NaN_cell_removal_with_mouseID(np.where(mod_cells)[0], mouseID)
		mod_cells_by_mouse[mouseID] = mod_cells
		
		# plot pre- vs post-drop performance for active grasp trained models
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			color='k', marker='.', alpha=0.3)
		plt.xlabel('pre-drop')
		plt.ylabel('post-drop')
		plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
		plt.title(f'{mouseID}')
		plt.xlim(-.05, 1)
		plt.ylim(-.05, 1)
		if splitter=='Shuffle':
			plt.savefig(fig_save_dir + f'pre vs post active {which_measure} r2 scatterplot trial split test {test_size}.png')
		elif splitter=='KFold':
			plt.savefig(fig_save_dir + f'pre vs post active {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
		if show_figs:
			plt.show()
		else:
			plt.close()

		# plot pre- vs post-drop performance for empty grasp trained models
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			color='k', marker='.', alpha=0.3)
		plt.xlabel('pre-drop')
		plt.ylabel('post-drop')
		plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
		plt.title(f'{mouseID}')
		plt.xlim(-.05, 1)
		plt.ylim(-.05, 1)
		if splitter=='Shuffle':
			plt.savefig(fig_save_dir + f'pre vs post empty {which_measure} r2 scatterplot trial split test {test_size}.png')
		elif splitter=='KFold':
			plt.savefig(fig_save_dir + f'pre vs post empty {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
		if show_figs:
			plt.show()
		else:
			plt.close()

		# plot pre- vs post-drop performance, coloration by which trial type models were trained on
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			color='b', marker='.', alpha=0.3, label='empty grasp')
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			color='r', marker='.', alpha=0.3, label='active grasp')
		# how many cells are predicted above 0 across mice
		print('empty cells r2>0: ', np.sum(np.logical_or(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))]>0, 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))]>0)))
		print('active cells r2>0: ', np.sum(np.logical_or(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))]>0, 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))]>0)))

		# plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells], 
		# 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells], 
		# 	color='c', marker='o')
		# plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells], 
		# 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells], 
		# 	color='m', marker='o')
		plt.legend()
		plt.xlabel('pre-drop')
		plt.ylabel('post-drop')
		plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
		plt.title(f'{mouseID}')
		plt.xlim(-.05, 1)
		plt.ylim(-.05, 1)
		if splitter=='Shuffle':
			plt.savefig(fig_save_dir + f'pre vs post compare couplings {which_measure} r2 scatterplot trial split test {test_size}.png')
		elif splitter=='KFold':
			plt.savefig(fig_save_dir + f'pre vs post compare couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
		if show_figs:
			plt.show()
		else:
			plt.close()

		# plot active vs empty trained model performance, coloration by pre- or post-drop testing
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			color='b', marker='.', alpha=0.3, label='pre-drop')
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))], 
			color='g', marker='.', alpha=0.3, label='post-drop')
		# plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells], 
		# 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells], 
		# 	color='c', marker='o')
		# plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells], 
		# 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells], 
		# 	color='lime', marker='o')
		plt.legend()
		plt.xlabel('active grasp')
		plt.ylabel('empty grasp')
		plt.plot([-1, 1], [-1, 1], '--', color='grey')
		plt.title(f'{mouseID}')
		plt.xlim(-.1, 1)
		plt.ylim(-.1, 1)
		if splitter=='Shuffle':
			plt.savefig(fig_save_dir + f'active vs empty compare couplings {which_measure} r2 scatterplot trial split test {test_size}.png')
		elif splitter=='KFold':
			plt.savefig(fig_save_dir + f'active vs empty compare couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
		if show_figs:
			plt.show()
		else:
			plt.close()

	# how many cells are predicted above 0 across mice
	print('empty cells r2>0: ', np.sum(np.logical_or(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')]>0, 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')]>0)))
	print('active cells r2>0: ', np.sum(np.logical_or(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')]>0, 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')]>0)))

	# scatterplot combined across mice, pre- vs post-drop, empty only
	plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
			color='k', marker='.', alpha=0.15)
	for mouseID in mice:
		# plot the location of modulated cells
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
			color='r', marker='.')
	result = np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')][:, np.newaxis], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')])
	plt.xlabel('pre-drop')
	plt.ylabel('post-drop')
	plt.plot([-1, 1], [-1, 1], '--', color='grey')
	plt.xlim(-.15, 1)
	plt.ylim(-.15, 1)
	plt.title(f'm={result[0][0]:.5f}, res={result[1][0]:.5f}')
	plt.suptitle('Encoding Performance, Pre- Vs Post-Drop, Empty Couplings')
	if splitter=='Shuffle':
		plt.savefig(multimouse_fig_dir + f'pre vs post empty couplings {which_measure} r2 scatterplot trial split test {test_size}.png')
	if splitter=='KFold':
		plt.savefig(multimouse_fig_dir + f'pre vs post empty couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
	if show_figs:
		plt.show()
	else:
		plt.close()

	# scatterplot combined across mice, pre vs post-drop, active only
	plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
			color='k', marker='.', alpha=0.15)
	for mouseID in mice:
		# plot the location of modulated cells
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
			color='r', marker='.')
	result = np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')][:, np.newaxis], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')])
	plt.xlabel('pre-drop')
	plt.ylabel('post-drop')
	plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
	plt.xlim(-.05, 1)
	plt.ylim(-.05, 1)
	plt.title(f'm={result[0][0]:.5f}, res={result[1][0]:.5f}')
	plt.suptitle('Encoding Performance, Pre- Vs Post-Drop, Active Couplings')
	if splitter=='Shuffle':
		plt.savefig(multimouse_fig_dir + f'pre vs post active couplings {which_measure} r2 scatterplot trial split test {test_size}.png')
	if splitter=='KFold':
		plt.savefig(multimouse_fig_dir + f'pre vs post active couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
	if show_figs:
		plt.show()
	else:
		plt.close()

	# scatterplot combined across mice
	# pre- vs post-drop, colored by active or empty trained
	fig, ax = plt.subplots()
	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
			color='r', marker='.', alpha=0.15, label='active grasp')
	active_res = np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')][:, np.newaxis], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')])
	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
			color='b', marker='.', alpha=0.15, label='empty grasp')
	empty_res= np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')][:, np.newaxis], 
	 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')])
	# for mouseID in mice:
	# 	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
	# 		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
	# 			color='m', marker='o')
	# 	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
	# 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
	# 		color='c', marker='o')

	ax.legend()
	ax.set_xlabel('pre-drop')
	ax.set_ylabel('post-drop')
	x = [-1, 0, 0, -1, -1]
	y = [-1, -1, 0, 0, -1]

	ax.fill(x, y, color='k', alpha=0.5)
	ax.plot([-1, 1], [-1, 1], '--', color='grey')
	
	ax.set_xlim(-.4, 1)
	ax.set_ylim(-.4, 1)
	ax.set_box_aspect(1)
	ax.set_title('Encoding Performance, Pre- Vs Post-Drop')
	if splitter=='Shuffle':
		fig.savefig(multimouse_fig_dir + f'pre vs post compare couplings {which_measure} r2 scatterplot trial split test {test_size}.pdf', dpi=550)
	if splitter=='KFold':
		fig.savefig(multimouse_fig_dir + f'pre vs post compare couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.pdf', dpi=550)
	if show_figs:
		plt.show()
	else:
		plt.close()
	print('active: ', active_res[0][0], active_res[1][0], ' empty: ', empty_res[0][0], empty_res[1][0])

	# scatterplot combined across mice
	# active vs empty trained, colored by applied on pre vs post-drop
	fig, ax = plt.subplots()
	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
			color='b', marker='.', alpha=0.2, label='pre-drop', rasterized=True)
	pre_res = np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active')][:, np.newaxis], 
	 	np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty')])
	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')], 
			color='g', marker='.', alpha=0.2, label='post-drop', rasterized=True)
	post_res = np.linalg.lstsq(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active')][:, np.newaxis], 
		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty')])
	# for mouseID in mice:
	# 	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
	# 		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='pre-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
	# 		color='c', marker='o')
	# 	ax.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='active'))][mod_cells_by_mouse[mouseID]], 
	# 		np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.logical_and(np.array(r2_dict['pre_post'], dtype=object)=='post-drop', np.array(r2_dict['category'], dtype=object)=='empty'))][mod_cells_by_mouse[mouseID]], 
	# 		color='lime', marker='o')

	x = [-1, 0, 0, -1, -1]
	y = [-1, -1, 0, 0, -1]

	ax.fill(x, y, color='k', alpha=0.5)
	ax.legend()
	ax.set_xlabel('active grasp')
	ax.set_ylabel('empty grasp')
	ax.plot([-1, 1], [-1, 1], '--', color='grey')
	ax.set_xlim(-.4, 1)
	ax.set_ylim(-.4, 1)
	ax.set_box_aspect(1)
	ax.set_title('Encoding Performance on Drops, Active vs Empty Couplings')
	if splitter=='Shuffle':
		plt.savefig(multimouse_fig_dir + f'active vs empty compare pre-post {which_measure} r2 scatterplot trial split test {test_size}.pdf', dpi=550)
	if splitter=='KFold':
		plt.savefig(multimouse_fig_dir + f'active vs empty compare pre-post {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.pdf', dpi=550)
	if show_figs:
		plt.show()
	else:
		plt.close()

	print('pre: ', pre_res[0][0], pre_res[1][0], ' post: ', post_res[0][0], post_res[1][0])
