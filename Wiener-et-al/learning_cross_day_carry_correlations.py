'''
day pair correlations 
'''
import numpy as np
import src.IO
import src.utils
from scipy import stats
import matplotlib.pyplot as plt

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

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
behaviors = [1, 2]
behavior_labels = ['active', 'empty']
pair_labels = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']

t_pre=4
t_post=4
calculate_PETHs=True
cross_mouse=True

exclude_low_count=False
low_count_thresh=4
late_m46=False
subsample_to_smaller=True

multi_mouse_save_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/learning/'

for mouseID in mice:
	print(mouseID)
	days = src.IO.get_days(mouseID)
	day_pairs = []
	for d, day in enumerate(days[:-1]):
		day_pairs.append([day, days[d+1]])
	print(day_pairs)
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	save_dir = mouse_dir + 'carry_analysis/learning/'
	if calculate_PETHs:
		correlations = [[[],[],[],[],[],[],[],[],[],[],[]] for beh in behaviors]
		pair_count = -1
		for pair in day_pairs:
			not_a_pair=False
			try:
				reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, pair)
			except:
				# accounts for shaping days before training starts which do not have aligned cell data
				continue

			F_multi = [[] for i in range(len(pair))]
			time_multi = [[[] for i in range(len(pair))] for i in range(len(behaviors))]
			carry_labels_multi = [[] for i in range(len(pair))]
			for i, day in enumerate(pair):
				s2p_fld = src.IO.get_s2p_fld(mouseID, day)
				f = np.load(s2p_fld + 'cascade_spks.npy')
				F_reg = f[reg_inds[i], :]
				n_neurons = F_reg.shape[0]
				try:
					event_times = np.load(s2p_fld + 'calcium_carry_times.npy')
					event_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
				except:
					if pair_count>0:
						correlations[0][pair_count] = np.zeros(n_neurons)*np.nan
						correlations[1][pair_count] = np.zeros(n_neurons)*np.nan
						pair_count+=1
					not_a_pair=True
					continue
				F_multi[i] = F_reg
				for b_class, beh in enumerate(behaviors):
					time_multi[b_class][i] = event_times[event_labels==beh]
			
			if not_a_pair:
				continue
			else:
				pair_count+=1

			# remove cells with NaNs
			NaN_indices = np.zeros(n_neurons, dtype=bool)
			for day_i, F in enumerate(F_multi):
				NaN = np.isnan(F).all(axis=1)
				NaN_indices = np.logical_or(NaN_indices, NaN)
			print(f'{np.sum(NaN_indices)} cells are NaN, removing...')
			n_neurons = np.sum(~NaN_indices)

			for day_i, F in enumerate(F_multi):
				F_multi[day_i] = F[~NaN_indices, :]

			if subsample_to_smaller:
				# find the smaller trial category on both days in the pair
				day_one_size = np.min([len(time_multi[0][0]), len(time_multi[1][0])])
				day_two_size = np.min([len(time_multi[0][1]), len(time_multi[1][1])])
				print(day_one_size, len(time_multi[0][0]), len(time_multi[1][0]), day_two_size, len(time_multi[0][1]), len(time_multi[1][1]))

			for b, beh in enumerate(behaviors):
				if exclude_low_count:
					# if the count on either day in the pair is below a threshold
					if len(time_multi[b][0])<low_count_thresh or len(time_multi[b][1])<low_count_thresh:
						correlations[b][pair_count] = np.zeros(n_neurons)*np.nan
						continue
				if subsample_to_smaller:
					repeats=100
					subsampled_corrs = np.zeros((n_neurons, repeats))*np.nan
					for repeat in range(repeats):
						# randomly subsample trials
						d1_times = np.random.choice(time_multi[b][0], size=day_one_size, replace=False)
						d2_times = np.random.choice(time_multi[b][1], size=day_two_size, replace=False)
						# calculate PETHs
						PETH_day_1_corr, _ = src.utils.plot_PETH(F_multi[0], d1_times, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None,norm=False)
						PETH_day_2_corr, _ = src.utils.plot_PETH(F_multi[1], d2_times, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None,norm=False)
						# find the correlation between the PETHs per cell
						cross_day_corr, _ = stats.pearsonr(PETH_day_1_corr, PETH_day_2_corr, axis=1)
						subsampled_corrs[:, repeat] = cross_day_corr
					correlations[b][pair_count] = np.nanmean(subsampled_corrs, axis=1)
				else:
					# calculate PETHs
					PETH_day_1_corr, _ = src.utils.plot_PETH(F_multi[0], time_multi[b][0], t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None,norm=False)
					PETH_day_2_corr, _ = src.utils.plot_PETH(F_multi[1], time_multi[b][1], t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None,norm=False)
					# correlate PETHs per cell
					cross_day_corr, _ = stats.pearsonr(PETH_day_1_corr, PETH_day_2_corr, axis=1)
					correlations[b][pair_count] = cross_day_corr
				print(f'{behavior_labels[b]} count: {len(time_multi[b][0])}, {len(time_multi[b][1])}')

		# save out correlations
		if subsample_to_smaller:
			if exclude_low_count:
				src.IO.save_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl', correlations)
			else:
				src.IO.save_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post.pkl', correlations)
		else:
			if exclude_low_count:
				src.IO.save_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl', correlations)
			else:
				src.IO.save_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post.pkl', correlations)
	else:
		# load in correlations
		if subsample_to_smaller:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post.pkl')
		else:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post.pkl')

	# plot with error bars correlations across days for each behavior
	colors = ['indianred', 'k']
	for b, beh in enumerate(behaviors):
		label = behavior_labels[b]
		average_correlation = np.zeros(len(correlations[b]))*np.nan
		corr_std_error = np.zeros(len(correlations[b]))*np.nan
		for pair in range(len(correlations[b])):
			average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(correlations[b][pair])))
			corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(correlations[b][pair]), nan_policy='omit'))
		plt.errorbar(pair_labels, average_correlation, yerr=corr_std_error, color=colors[b], ecolor=colors[b], capsize=3, label=label)
	plt.title(f'{mouseID} Paired PETH Correlations Over Time')
	plt.legend()
	if subsample_to_smaller:
		if exclude_low_count:
			plt.savefig(save_dir + f'Paired PETH Correlations Over Time subsampled {t_pre} pre {t_post} post low count excluded.png')
		else:
			plt.savefig(save_dir + f'Paired PETH Correlations Over Time subsampled {t_pre} pre {t_post} post.png')
	else:	
		if exclude_low_count:
			plt.savefig(save_dir + f'Paired PETH Correlations Over Time {t_pre} pre {t_post} post low count excluded.png')
		else:
			plt.savefig(save_dir + f'Paired PETH Correlations Over Time {t_pre} pre {t_post} post.png')
	plt.close()

if cross_mouse:
	cross_mouse_correlations = [[[],[],[],[],[],[],[],[],[],[],[]] for beh in behaviors]
	for mouseID in mice:
		print(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/learning/'
		if subsample_to_smaller:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post.pkl')
		else:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post.pkl')
		for b in range(len(behaviors)):
			for pair in range(len(correlations[b])):
				cross_mouse_correlations[b][pair].extend(correlations[b][pair])

	# plot with error bars correlations across days for each behavior combined across mice
	colors = ['indianred', 'k']
	fig, ax = plt.subplots()
	for b, beh in enumerate(behaviors):
		label = behavior_labels[b]
		average_correlation = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		corr_std_error = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		for pair in range(len(cross_mouse_correlations[b])):
			average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(cross_mouse_correlations[b][pair])))
			corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(cross_mouse_correlations[b][pair]), nan_policy='omit'))
		ax.errorbar(pair_labels, average_correlation, yerr=corr_std_error, color=colors[b], ecolor=colors[b], capsize=3, label=label)
	ax.set_title(f'PETH Stability Over Time')
	ax.set_xlabel('Training Day Pair')
	ax.set_ylabel('Correlation')
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	if subsample_to_smaller:
		if exclude_low_count:
			fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled {t_pre} pre {t_post} post exclude low count.png', dpi=600)
		else:
			fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled {t_pre} pre {t_post} post.png', dpi=600)
	else:
		if exclude_low_count:
			fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time {t_pre} pre {t_post} post exclude low count.png', dpi=600)
		else:
			fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time {t_pre} pre {t_post} post.png', dpi=600)
	plt.show()


	cross_mouse_correlations = [[[] for a in np.arange(-7, 10)] for beh in behaviors]
	pairs = np.arange(-7, 10)
	pair_labels = ['-7,-6','-6,-5', '-5,-4', '-4,-3', '-3,-2', '-2,-1', '-1,0', '0,1', '1,2', '2,3', '3,4', '4,5', '5,6', '6,7', '7,8', '8,9', '9,10']
	for mouseID in mice:
		print(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/learning/'
		if subsample_to_smaller:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post.pkl')
		else:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post.pkl')
		transition_idx = get_relative_crossover_pair(mouseID, late_m46)
		for b in range(len(behaviors)):
			for pair in range(len(correlations[b])):
				p = pair-transition_idx+7
				print(p)
				if p>(len(pairs)-1):
					continue
				cross_mouse_correlations[b][p].extend(correlations[b][pair])

	# plot with error bars correlations across days for each behavior combined across mice
	colors = ['indianred', 'k']
	fig, ax = plt.subplots()
	max_options = []
	for b, beh in enumerate(behaviors):
		label = behavior_labels[b]
		average_correlation = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		corr_std_error = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		for pair in range(len(cross_mouse_correlations[b])):
			average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(cross_mouse_correlations[b][pair])))
			corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(cross_mouse_correlations[b][pair]), nan_policy='omit'))
		ax.plot(pairs, average_correlation, color=colors[b], label=label)
		ax.fill_between(pairs, average_correlation+corr_std_error, average_correlation-corr_std_error, color=colors[b], alpha=0.4)
		max_options.append(np.nanmax(average_correlation+corr_std_error))
	ax.axvline(-.5, ls='--', color='k')
	ax.set_title(f'PETH Stability Over Time')
	ax.set_xlabel('Training Day Pair Relative to Crossover')
	ax.set_ylabel('Correlation')
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlim([-5, 5])
	ax.set_ylim([0, np.max(max_options)])
	ax.set_xticks(np.arange(-5, 6), ['(-5,-4)', '(-4,-3)', '(-3,-2)', '(-2,-1)', '(-1,0)', '(0,1)', '(1,2)', '(2,3)', '(3,4)', '(4,5)', '(5,6)'])
	if subsample_to_smaller:
		if late_m46:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Crossover low count excluded late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Crossover late m46.png')
		else:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Crossover low count excluded.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Crossover.pdf', dpi=550)
	else:
		if late_m46:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Crossover low count excluded late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Crossover late m46.png')
		else:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Crossover low count excluded.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Crossover.png')
	plt.show()

	cross_mouse_correlations = [[[] for a in np.arange(-7, 9)] for beh in behaviors]
	pairs = np.arange(-7, 9)
	pair_labels = ['-7,-6','-6,-5', '-5,-4', '-4,-3', '-3,-2', '-2,-1', '-1,0', '0,1', '1,2', '2,3', '3,4', '4,5', '5,6', '6,7', '7,8', '8,9']
	for mouseID in mice:
		print(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/learning/'
		if subsample_to_smaller:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_subsampled_{t_pre}_pre_{t_post}_post.pkl')
		else:
			if exclude_low_count:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post_exclude_low_count.pkl')
			else:
				correlations = src.IO.load_pickle(save_dir + f'day_pair_correlations_{t_pre}_pre_{t_post}_post.pkl')
		transition_idx = get_relative_peak_pair(mouseID, late_m46)
		for b in range(len(behaviors)):
			for pair in range(len(correlations[b])):
				p = pair-transition_idx+abs(pairs[0])
				if p>(len(pairs)-1):
					continue
				cross_mouse_correlations[b][p].extend(correlations[b][pair])

	# plot with error bars correlations across days for each behavior combined across mice
	colors = ['indianred', 'k']
	fig, ax = plt.subplots()
	for b, beh in enumerate(behaviors):
		label = behavior_labels[b]
		average_correlation = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		corr_std_error = np.zeros(len(cross_mouse_correlations[b]))*np.nan
		for pair in range(len(cross_mouse_correlations[b])):
			average_correlation[pair] = np.tanh(np.nanmean(np.arctanh(cross_mouse_correlations[b][pair])))
			corr_std_error[pair] = np.tanh(stats.sem(np.arctanh(cross_mouse_correlations[b][pair]), nan_policy='omit'))
		ax.plot(pairs, average_correlation, color=colors[b], label=label)
		ax.fill_between(pairs, average_correlation+corr_std_error, average_correlation-corr_std_error, color=colors[b], alpha=0.4)
	ax.axvline(-.5, ls='--', color='k')
	ax.set_title(f'PETH Stability Over Time')
	ax.set_xlabel('Training Day Pair Relative to Empty Peak')
	ax.set_ylabel('Correlation')
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlim([-3, 6])
	ax.set_xticks(np.arange(-3, 7), ['(-3,-2)', '(-2,-1)', '(-1,0)', '(0,1)', '(1,2)', '(2,3)', '(3,4)', '(4,5)', '(5,6)', '(6,7)'])
	if subsample_to_smaller:
		if late_m46:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Empty Peak low count excluded late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Empty Peak late m46.png')
		else:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Empty Peak low count excluded.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time subsampled Aligned to Empty Peak.png')
	else:
		if late_m46:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Empty Peak low count excluded late m46.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Empty Peak late m46.png')
		else:
			if exclude_low_count:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Empty Peak low count excluded.png')
			else:
				fig.savefig(multi_mouse_save_dir + f'Paired PETH Correlations Over Time Aligned to Empty Peak.png')
	plt.show()