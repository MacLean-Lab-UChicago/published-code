'''
pairs of day correlations and single day correlations of PETHs
'''
import numpy as np
import src.IO
import src.utils
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
data_dir = Path.cwd() / 'data'

def get_relative_peak_day(mouseID, late_m46):
	if mouseID=='mouse22':
		peak = 0
	elif mouseID=='mouse25':
		peak = 3
	elif mouseID=='mouse39':
		peak = 2
	elif mouseID=='mouse35':
		peak = 2
	elif mouseID=='mouse46':
		if late_m46:
			peak = 7
		else:
			peak = 2
	elif mouseID=='mouse51':
		peak = 3
	elif mouseID=='mouse549':
		peak = 2
	else:
		peak=None
	return peak

def get_relative_crossover_day(mouseID, late_m46):
	if mouseID=='mouse22':
		day = 1
	elif mouseID=='mouse25':
		day = 5
	elif mouseID=='mouse39':
		day = 3
	elif mouseID=='mouse35':
		day = 6
	elif mouseID=='mouse46':
		if late_m46:
			day = 8
		else:
			day = 4
	elif mouseID=='mouse51':
		day = 5
	elif mouseID=='mouse549':
		day = 6
	else:
		day=None
	return day

def get_relative_crossover_pair(mouseID, late_m46):
	if mouseID=='mouse22':
		peak = 0
	elif mouseID=='mouse25':
		peak = 4
	elif mouseID=='mouse39':
		peak = 2
	elif mouseID=='mouse35':
		peak = 5
	elif mouseID=='mouse549':
		peak = 5
	elif mouseID=='mouse51':
		peak = 4
	elif mouseID=='mouse46':
		if late_m46:
			peak = 7
		else:
			peak = 3
	else:
		peak=None
	return peak

def get_relative_peak_pair(mouseID, late_m46):
	peak = get_relative_peak_day(mouseID, late_m46)
	return peak

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
t_pre = 4 # frames
t_post = 4 # frames
time_in_trial=t_pre+t_post # frames
bootstrap_count = 1000
carry_classes = [1, 2]
labels = ['active', 'empty']
training_days = np.arange(12)+1
pair_labels = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']
multi_mouse_save_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/learning/'

calculate_correlations = True
cross_mouse = True
day_pairs = True
omit_low_count = False
different_m46 = False
subsample_to_smaller = False
low_count_thresh = 10

if calculate_correlations:
	if day_pairs:
		for subsample_to_smaller in [True, False]:
			for mouseID in mice:
				print(mouseID)
				days = src.IO.get_days(mouseID)
				day_pairs = []
				for d, day in enumerate(days[:-1]):
					day_pairs.append([day, days[d+1]])
				print(day_pairs)
				drive = data_dir + '/neural'
				mouse_dir = drive + '/' + mouseID + '/'
				save_dir = mouse_dir + 'carry_analysis/learning/'
				
				correlations = [[[],[],[],[],[],[],[],[],[],[],[]] for c in carry_classes]
				pair_count = -1
				for pair in day_pairs:
					not_a_pair=False
					try:
						reg_inds = src.utils.load_registered_cells(mouse_dir, pair)
					except:
						# to exclude days in the data during shaping (prior to training days)
						continue

					F_multi = [[] for i in range(len(pair))]
					time_multi = [[[] for i in range(len(pair))] for i in range(len(carry_classes))]
					carry_labels_multi = [[] for i in range(len(pair))]
					for i, day in enumerate(pair):
						s2p_fld = src.IO.get_s2p_fld(mouseID, day)
						# load in neural activity
						f = np.load(s2p_fld + 'cascade_spks.npy')
						# filter to cells present on both days
						F_reg = f[reg_inds[i], :]
						n_neurons = F_reg.shape[0]
						try:
							event_times = np.load(s2p_fld + 'calcium_carry_times.npy')
							event_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
						except:
							if pair_count>0: # if a training day with no carry labels or times (occurs if no carry instances)
								correlations[0][pair_count] = np.zeros(n_neurons)*np.nan
								correlations[1][pair_count] = np.zeros(n_neurons)*np.nan
								pair_count+=1
							not_a_pair=True
							continue
						F_multi[i] = F_reg
						for c, carry in enumerate(carry_classes):
							time_multi[c][i] = event_times[event_labels==carry]
					
					if not_a_pair:
						continue
					else:
						pair_count+=1

					# remove any cells that lack inferred spike counts on either day
					NaN_indices = np.zeros(n_neurons, dtype=bool)
					for day_i, F in enumerate(F_multi):
						NaN = np.isnan(F).all(axis=1)
						NaN_indices = np.logical_or(NaN_indices, NaN)
					print(f'{np.sum(NaN_indices)} cells are NaN, removing...')
					n_neurons = np.sum(~NaN_indices)

					for day_i, F in enumerate(F_multi):
						F_multi[day_i] = F[~NaN_indices, :]

					if subsample_to_smaller:
						# find split size as half of the smaller class on that pair of days
						split_size = int(.5*np.min([len(time_multi[0][0])+len(time_multi[0][1]), len(time_multi[1][0])+len(time_multi[1][1])]))

					for c in range(len(carry_classes)):
						bootstrapped_coors = []
						if omit_low_count:
							# if the number of trials on this pair of days is too low
							if (len(time_multi[c][0])+len(time_multi[c][1]))<low_count_thresh:
								correlations[c][pair_count] = np.zeros(n_neurons)*np.nan
								continue
						# convert multiday into something that can be indexed by a random choice
						times = time_multi[c]
						cross_day_times = []
						for day_i in range(len(pair)):
							cross_day_times.extend([f'{day_i}_{int(time)}' for time in times[day_i]])
						for repeat in range(bootstrap_count):
							if subsample_to_smaller:
								# randomly split the data in half
								split = np.random.choice(len(cross_day_times), size=split_size, replace=False)
								sec_half = np.random.choice([i for i in range(len(cross_day_times)) if i not in split], size=split_size, replace=False)
								# restore the multiday nested list from the split indices
								carry_times_1half = [[] for i in range(len(pair))]
								carry_times_2half = [[] for i in range(len(pair))]
								for time in cross_day_times:
									day = int(time.split('_')[0])
									t_stamp = int(time.split('_')[1])
									if time in np.array(cross_day_times, dtype=object)[split]:
										carry_times_1half[day].append(t_stamp)
									elif time in np.array(cross_day_times, dtype=object)[sec_half]:
										carry_times_2half[day].append(t_stamp)
							else:
								# randomly split the data in half
								split = np.random.choice(len(cross_day_times), size=int(.5*len(cross_day_times)), replace=False)
								# restore the multiday nested list from the split indices
								carry_times_1half = [[] for i in range(len(pair))]
								carry_times_2half = [[] for i in range(len(pair))]
								for time in cross_day_times:
									day = int(time.split('_')[0])
									t_stamp = int(time.split('_')[1])
									if time in np.array(cross_day_times, dtype=object)[split]:
										carry_times_1half[day].append(t_stamp)
									else:
										carry_times_2half[day].append(t_stamp)
							for day_i in range(len(pair)):
								carry_times_1half[day_i] = np.array(carry_times_1half[day_i])
								carry_times_2half[day_i] = np.array(carry_times_2half[day_i])

							# calculate PETHs
							half1_PETH, ind = src.utils.plot_PETH_multiday(pair, F_multi, carry_times_1half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)
							half2_PETH, ind = src.utils.plot_PETH_multiday(pair, F_multi, carry_times_2half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)

							# find correlations and p-values between PETHs
							corr, pval = stats.pearsonr(half1_PETH, half2_PETH, axis=1)
							bootstrapped_coors.append(corr)
						# average (with a z-transform) across bootstraps
						correlations[c][pair_count] = np.tanh(np.nanmean(np.arctanh(bootstrapped_coors), axis=0))
				
				# save out calculations
				if subsample_to_smaller:
					if omit_low_count:
						src.IO.save_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped_low_count_omitted_{low_count_thresh}.pkl', correlations)
					else:
						src.IO.save_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped.pkl', correlations)
				else:
					if omit_low_count:
						src.IO.save_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl', correlations)
					else:
						src.IO.save_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_bootstrapped.pkl', correlations)
	else:
		for mouseID in mice:
			days = src.IO.get_days(mouseID)
			drive = data_dir + '/neural'
			mouse_dir = drive + '/' + mouseID + '/'
			save_dir = mouse_dir + 'carry_analysis/learning/'
			correlations = [[[] for day in training_days] for c in carry_classes]
			day_counter = -1
			for day in days:
				s2p_fld = src.IO.get_s2p_fld(mouseID, day)
				try:
					f = np.load(s2p_fld + 'cascade_spks.npy')
				except:
					if day_counter>0:
						day_counter+=1
						correlations[0][day_counter]=np.zeros(n_neurons)*np.nan
						correlations[1][day_counter]=np.zeros(n_neurons)*np.nan
					continue
				n_neurons = f.shape[0]
				try:
					carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
					carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
					day_counter+=1
				except:
					if day_counter>0:
						day_counter+=1
						correlations[0][day_counter]=np.zeros(n_neurons)*np.nan
						correlations[1][day_counter]=np.zeros(n_neurons)*np.nan
					continue
			
				NaN_indices = np.isnan(f).all(axis=1)
				print(f'{np.sum(NaN_indices)} cells are NaN, removing...')
				n_neurons = np.sum(~NaN_indices)
				F = f[~NaN_indices, :]

				if subsample_to_smaller:
					split_size = int(.5*(np.min([np.sum(carry_labels==1), np.sum(carry_labels==2)])))
					print(split_size, np.sum(carry_labels==1), np.sum(carry_labels==2))
					if split_size==0:
						correlations[0][day_counter]=np.zeros(n_neurons)*np.nan
						correlations[1][day_counter]=np.zeros(n_neurons)*np.nan
						continue

				for c, carry_type in enumerate(carry_classes):
					# convert multiday into something that can be indexed by a random choice
					times = carry_times[carry_labels==carry_type]
					bootstrapped_coors = []
					if omit_low_count:
						if len(times)<low_count_thresh:
							bootstrapped_coors = np.zeros((bootstrap_count, n_neurons))*np.nan
							continue
					for repeat in range(bootstrap_count):
						if subsample_to_smaller:
							# randomly split the data in half
							split = np.random.choice(len(times), size=split_size, replace=False)
							sec_half = np.random.choice([i for i in range(len(times)) if i not in split], size=split_size, replace=False)
							# restore the multiday nested list from the split indices
							carry_times_1half = times[split]
							carry_times_2half = times[sec_half]
						else:
							# randomly split the data in half
							split = np.random.choice(len(times), size=int(.5*len(times)), replace=False)
							# restore the multiday nested list from the split indices
							carry_times_1half = times[split]
							carry_times_2half = np.delete(times, split)
						
						half1_PETH, ind = src.utils.plot_PETH(F, carry_times_1half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)
						half2_PETH, ind = src.utils.plot_PETH(F, carry_times_2half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)

						# find correlations and p-values
						corr, pval = stats.pearsonr(half1_PETH, half2_PETH, axis=1)
						bootstrapped_coors.append(corr)
					correlations[c][day_counter] = np.tanh(np.nanmean(np.arctanh(bootstrapped_coors), axis=0))
			if subsample_to_smaller:
				if omit_low_count:
					src.IO.save_pickle(save_dir + f'within_day_per_day_PETH_correlations_subsampled_bootstrapped_low_count_omitted_{low_count_thresh}.pkl', correlations)
				else:
					src.IO.save_pickle(save_dir + 'within_day_per_day_PETH_correlations_subsampled_bootstrapped.pkl', correlations)
			else:
				if omit_low_count:
					src.IO.save_pickle(save_dir + f'within_day_per_day_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl', correlations)
				else:
					src.IO.save_pickle(save_dir + 'within_day_per_day_PETH_correlations_bootstrapped.pkl', correlations)
# plotting!
if day_pairs:
	for subsample_to_smaller in [True, False]:
		training_day_pairs = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']
		cross_mouse_correlations = [[[] for day in training_day_pairs] for c in carry_classes]
		for mouseID in mice:
			drive = data_dir + '/neural'
			mouse_dir = drive + '/' + mouseID + '/'
			save_dir = mouse_dir + 'carry_analysis/learning/'
			if subsample_to_smaller:
				if omit_low_count:
					correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
				else:
					correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped.pkl')
			else:
				if omit_low_count:
					correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
				else:
					correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_bootstrapped.pkl')

			colors = ['indianred', 'k']
			for c, carry_type in enumerate(carry_classes):
				label = labels[c]
				average_correlation = np.zeros(len(correlations[c]))*np.nan
				corr_std_error = np.zeros(len(correlations[c]))*np.nan
				for pair in range(len(correlations[c])):
					average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(np.clip(correlations[c][pair], -1 + 1e-15, 1 - 1e-15))))
					corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(np.clip(correlations[c][pair], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
					cross_mouse_correlations[c][pair].extend(correlations[c][pair])
				plt.errorbar(training_day_pairs, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
			plt.title(f'{mouseID} PETH Correlations Over Time')
			plt.xlabel('Training Day')
			plt.ylabel('Correlation')
			plt.legend()
			if subsample_to_smaller:
				if omit_low_count:
					plt.savefig(save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post subsampled omit low count {low_count_thresh}.png')
				else:
					plt.savefig(save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post subsampled.png')
			else:
				if omit_low_count:
					plt.savefig(save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post omit low count {low_count_thresh}.png')
				else:
					plt.savefig(save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post.png')
			plt.close()


		if cross_mouse:
			# cross-mouse plotting
			fig, ax = plt.subplots()
			for c, carry_type in enumerate(carry_classes):
				label = labels[c]
				average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				for pair in range(len(cross_mouse_correlations[c])):
					average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15))))
					corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
				ax.errorbar(training_day_pairs, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
			ax.set_title('Within-Day PETH Correlations Over Time')
			ax.set_xlabel('Training Day Pair')
			ax.set_ylabel('Correlation')
			ax.legend()
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			if subsample_to_smaller:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post subsampled omit low count {low_count_thresh}.png', dpi=600)
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post subsampled.png', dpi=600)
			else:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post omit low count {low_count_thresh}.png', dpi=600)
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time {t_pre} pre {t_post} post.png', dpi=600)
			plt.show()

			# same but aligning to peak day
			pair_rel_to_peak = np.arange(-8, 10)
			cross_mouse_correlations = [[[] for a in pair_rel_to_peak] for c in carry_classes]
			for mouseID in mice:
				save_dir = src.IO.get_drive(mouseID) + '/' + mouseID + '/carry_analysis/learning/'
				if subsample_to_smaller:
					if omit_low_count:
						correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
					else:
						correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped.pkl')
				else:
					if omit_low_count:
						correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
					else:
						correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_bootstrapped.pkl')

				# align each mouse such that the 8th index represents the empty peak day
				peak_pair = get_relative_peak_pair(mouseID, different_m46)
				for c, carry_type in enumerate(carry_classes):
					for pair in range(len(correlations[c])):
						d = 8-peak_pair+pair
						if d>(len(pair_rel_to_peak)-1):
							continue
						cross_mouse_correlations[c][d].extend(correlations[c][pair])

			pair_labels = ['-8,-7', '-7,-6', '-6,-5', '-5,-4', '-4,-3', '-3,-2', '-2,-1', '-1,0', '0,1', '1,2', '2,3', '3,4', '4,5', '5,6', '6,7', '7,8', '8,9', '9,10']
			fig, ax = plt.subplots()
			for c, carry in enumerate(carry_classes):
				label = labels[c]
				average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				for pair in range(len(cross_mouse_correlations[c])):
					average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15))))
					corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
				ax.errorbar(pair_labels, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
			ax.set_title('Within-Day PETH Correlations Over Time')
			ax.set_xlabel('Training Day Pair Relative to Empty Peak')
			ax.set_ylabel('Correlation')
			ax.legend()
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			if subsample_to_smaller:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time subsampled Aligned to Empty Peak low count excluded {low_count_thresh}.png')
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time subsampled Aligned to Empty Peak.png')
			elif different_m46:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Empty Peak low count excluded {low_count_thresh} late m46.png')
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Empty Peak late m46.png')
			else:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Empty Peak low count excluded {low_count_thresh}.png')
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Empty Peak.png')
			plt.show()
			
			# same but aligning to crossover day pair
			pair_rel_to_peak = np.arange(-9, 10)
			cross_mouse_correlations = [[[] for a in pair_rel_to_peak] for c in carry_classes]
			for mouseID in mice:
				save_dir = src.IO.get_drive(mouseID) + '/' + mouseID + '/carry_analysis/learning/'
				if subsample_to_smaller:
					if omit_low_count:
						correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
					else:
						correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped.pkl')
				else:
					if omit_low_count:
						correlations = src.IO.load_pickle(save_dir + f'within_day_pair_per_pair_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
					else:
						correlations = src.IO.load_pickle(save_dir + 'within_day_pair_per_pair_PETH_correlations_bootstrapped.pkl')

				# align for each mouse such that the 9th index is always the crossover day
				peak_pair = get_relative_crossover_pair(mouseID, different_m46)
				for c, carry_type in enumerate(carry_classes):
					for pair in range(len(correlations[c])):
						d = 9-peak_pair+pair
						if d>(len(pair_rel_to_peak)-1):
							continue
						cross_mouse_correlations[c][d].extend(correlations[c][pair])

			pair_labels = ['-9,-8', '-8,-7', '-7,-6', '-6,-5', '-5,-4', '-4,-3', '-3,-2', '-2,-1', '-1,0', '0,1', '1,2', '2,3', '3,4', '4,5', '5,6', '6,7', '7,8', '8,9', '9,10']
			fig, ax = plt.subplots()
			max_options = []
			for c, carry in enumerate(carry_classes):
				label = labels[c]
				average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
				for pair in range(len(cross_mouse_correlations[c])):
					average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15))))
					corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][pair], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
				ax.plot(pair_rel_to_peak, average_correlation, color=colors[c], label=label)
				ax.fill_between(pair_rel_to_peak, average_correlation+corr_std_error, average_correlation-corr_std_error, color=colors[c], alpha=0.4)
				max_options.append(np.nanmax(average_correlation+corr_std_error))
			ax.axvline(-.5, color='k', ls='--')
			ax.set_title('Within-Day PETH Correlations Over Time')
			ax.set_xlabel('Training Day Relative to Active-Empty Crossover')
			ax.set_ylabel('Correlation')
			ax.legend()
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.set_xlim([-5, 5])
			ax.set_ylim([0, np.max(max_options)])
			ax.set_xticks(np.arange(-5, 6), ['(-5,-4)', '(-4,-3)', '(-3,-2)', '(-2,-1)', '(-1,0)', '(0,1)', '(1,2)', '(2,3)', '(3,4)', '(4,5)', '(5,6)'])
			if subsample_to_smaller:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time subsampled Aligned to Crossover low count excluded {low_count_thresh}.pdf', dpi=660)
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time subsampled Aligned to Crossover.pdf', dpi=660)
			elif different_m46:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Crossover low count excluded {low_count_thresh} late m46.pdf', dpi=660)
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Crossover late m46.pdf', dpi=660)
			else:
				if omit_low_count:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Crossover low count excluded {low_count_thresh}.pdf', dpi=660)
				else:
					fig.savefig(multi_mouse_save_dir + f'Paired Days PETH Correlations Over Time Aligned to Crossover.pdf', dpi=660)
			plt.show()

else:
	cross_mouse_correlations = [[[] for day in training_days] for c in carry_classes]
	for mouseID in mice:
		drive = data_dir + '/neural'
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/learning/'
		if omit_low_count:
			correlations = src.IO.load_pickle(save_dir + f'within_day_per_day_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
		else:
			correlations = src.IO.load_pickle(save_dir + 'within_day_per_day_PETH_correlations_bootstrapped.pkl')
		# plot with error bars correlations across days for each behavior
		colors = ['indianred', 'k']
		for c, carry_type in enumerate(carry_classes):
			label = labels[c]
			average_correlation = np.zeros(len(correlations[c]))*np.nan
			corr_std_error = np.zeros(len(correlations[c]))*np.nan
			for day in range(len(correlations[c])):
				average_correlation[day] = np.tanh(np.nanmean(np.arctanh(np.clip(correlations[c][day], -1 + 1e-15, 1 - 1e-15))))
				corr_std_error[day] = np.tanh(stats.sem(np.arctanh(np.clip(correlations[c][day], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
				cross_mouse_correlations[c][day].extend(correlations[c][day])
			plt.errorbar(training_days, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
		plt.title(f'{mouseID} PETH Correlations Over Time')
		plt.xlabel('Training Day')
		plt.ylabel('Correlation')
		plt.legend()
		if subsample_to_smaller:
			if omit_low_count:
				plt.savefig(save_dir + f'Single Day PETH Correlations Over Time subsampled {t_pre} pre {t_post} post omit low count {low_count_thresh}.png', dpi=660)
			else:
				plt.savefig(save_dir + f'Single Day PETH Correlations Over Time subsampled {t_pre} pre {t_post} post.png')
		else:
			if omit_low_count:
				plt.savefig(save_dir + f'Single Day PETH Correlations Over Time {t_pre} pre {t_post} post omit low count {low_count_thresh}.png')
			else:
				plt.savefig(save_dir + f'Single Day PETH Correlations Over Time {t_pre} pre {t_post} post.png')
		plt.close()

	if cross_mouse:
		# cross-mouse plotting
		fig, ax = plt.subplots()
		for c, carry_type in enumerate(carry_classes):
			label = labels[c]
			average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			for day in range(len(cross_mouse_correlations[c])):
				average_correlation[day] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15))))
				corr_std_error[day] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
			print(average_correlation)
			ax.errorbar(training_days, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
		ax.set_title('Within-Day PETH Correlations Over Time')
		ax.set_xlabel('Training Day')
		ax.set_ylabel('Correlation')
		ax.legend()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		if subsample_to_smaller:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled {t_pre} pre {t_post} post omit low count {low_count_thresh}.png', dpi=600)
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled {t_pre} pre {t_post} post.png', dpi=600)
		else:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time {t_pre} pre {t_post} post omit low count {low_count_thresh}.png', dpi=600)
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time {t_pre} pre {t_post} post.png', dpi=600)
		plt.show()

		# same but aligning to peak day
		day_rel_to_peak = np.arange(-8, 10)
		cross_mouse_correlations = [[[] for a in day_rel_to_peak] for c in carry_classes]
		for mouseID in mice:
			save_dir = src.IO.get_drive(mouseID) + '/' + mouseID + '/carry_analysis/learning/'
			if omit_low_count:
				correlations = src.IO.load_pickle(save_dir + f'within_day_per_day_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + 'within_day_per_day_PETH_correlations_bootstrapped.pkl')

			peak_day = get_relative_peak_day(mouseID, different_m46)
			for c, carry_type in enumerate(carry_classes):
				for day in range(len(correlations[c])):
					d = 8-peak_day+day
					if d>(len(day_rel_to_peak)-1):
						continue
					cross_mouse_correlations[c][d].extend(correlations[c][day])

		fig, ax = plt.subplots()
		for c, carry in enumerate(carry_classes):
			label = labels[c]
			average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			for day in range(len(cross_mouse_correlations[c])):
				average_correlation[day] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15))))
				corr_std_error[day] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
			ax.errorbar(day_rel_to_peak, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
		ax.set_title('Within-Day PETH Correlations Over Time')
		ax.set_xlabel('Training Day Relative to Empty Peak')
		ax.set_ylabel('Correlation')
		ax.legend()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		if subsample_to_smaller:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled Aligned to Empty Peak low count excluded {low_count_thresh} late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled Aligned to Empty Peak late m46.png')
		elif different_m46:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Empty Peak low count excluded {low_count_thresh} late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Empty Peak late m46.png')
		else:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Empty Peak low count excluded {low_count_thresh}.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Empty Peak.png')
		plt.show()
		
		# same but aligning to Crossover day
		day_rel_to_peak = np.arange(-9, 10)
		cross_mouse_correlations = [[[] for a in day_rel_to_peak] for c in carry_classes]
		for mouseID in mice:
			save_dir = src.IO.get_drive(mouseID) + '/' + mouseID + '/carry_analysis/learning/'
			if omit_low_count:
				correlations = src.IO.load_pickle(save_dir + f'within_day_per_day_PETH_correlations_bootstrapped_low_count_omitted_{low_count_thresh}.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + 'within_day_per_day_PETH_correlations_bootstrapped.pkl')

			peak_day = get_relative_crossover_day(mouseID, different_m46)
			for c, carry_type in enumerate(carry_classes):
				for day in range(len(correlations[c])):
					d = 9-peak_day+day
					if d>(len(day_rel_to_peak)-1):
						continue
					cross_mouse_correlations[c][d].extend(correlations[c][day])

		fig, ax = plt.subplots()
		for c, carry in enumerate(carry_classes):
			label = labels[c]
			average_correlation = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			corr_std_error = np.zeros(len(cross_mouse_correlations[c]))*np.nan
			for day in range(len(cross_mouse_correlations[c])):
				average_correlation[day] = np.tanh(np.nanmean(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15))))
				corr_std_error[day] = np.tanh(stats.sem(np.arctanh(np.clip(cross_mouse_correlations[c][day], -1 + 1e-15, 1 - 1e-15)), nan_policy='omit'))
			ax.errorbar(day_rel_to_peak, average_correlation, yerr=corr_std_error, color=colors[c], ecolor=colors[c], capsize=3, label=label)
		ax.set_title('Within-Day PETH Correlations Over Time')
		ax.set_xlabel('Training Day Relative to Active-Empty Crossover')
		ax.set_ylabel('Correlation')
		ax.legend()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		if subsample_to_smaller:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled Aligned to Crossover low count excluded {low_count_thresh} late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time subsampled Aligned to Crossover late m46.png')
		elif different_m46:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Crossover low count excluded {low_count_thresh} late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Crossover late m46.png')
		else:
			if omit_low_count:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Crossover low count excluded {low_count_thresh}.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Single Day PETH Correlations Over Time Aligned to Crossover.png')
		plt.show()


