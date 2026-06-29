'''
single trial autocorrelations of specific cells
all pairwise combinations of within-category and cross-category trials
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import src.IO
import src.utils
import itertools

show_figs = True

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
t_pre = 4 # frames
t_post = 4 # frames
time_in_trial = t_pre + t_post

all_mice_within_cat = []
all_mice_cross_cat = []
multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'

for mouseID in mice:
	days = src.IO.get_carry_days(mouseID)
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	save_dir = mouse_dir + 'carry_analysis/figures/cell_PETHs/'
	
	reg_inds = src.utils.load_registered_cells(mouse_dir, days)
	neural_activity_by_trial = [[] for day in days]
	trial_labels = [[] for day in days]
	for i, day in enumerate(days):
		print(day)
		# load in the Cascade spikes
		s2p_fld = src.IO.get_s2p_fld(mouseID, day)		
		spks = np.load(s2p_fld + 'cascade_spks.npy')
		
		# index by the registered cells
		reg_spks = spks[reg_inds[i], :]
		n_neurons = reg_spks.shape[0]

		# load in the behavioral times
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		carry_times = carry_times[carry_labels!=0]
		carry_labels = carry_labels[carry_labels!=0]
		trial_labels[i] = carry_labels
		
		# pull out the relevant neural data
		temp_spks = np.zeros((n_neurons, len(carry_times), time_in_trial))
		for carry_i, carry in enumerate(carry_times.astype(int)):
			temp_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		neural_activity_by_trial[i] = temp_spks
	neural_activity_by_trial = np.concatenate(neural_activity_by_trial, axis=1)
	trial_labels = np.concatenate(trial_labels)

	# get active and empty trial indices
	trial_indices_active = np.where(trial_labels==1)[0]
	trial_indices_empty = np.where(trial_labels==2)[0]
	print(len(trial_indices_active), len(trial_indices_empty))

	# remove cells with NaNs
	print(n_neurons)
	NaN_cells = np.isnan(neural_activity_by_trial).any(axis=(1, 2))
	neural_activity_by_trial = neural_activity_by_trial[~NaN_cells, :, :]
	n_neurons = np.sum(~NaN_cells)
	print(n_neurons)

	# for all pairs of trials within categories
	# then find the correlation by cell
	same_same_correlations = []
	for i, j in itertools.permutations(trial_indices_active, 2):
		corr, _ = stats.pearsonr(neural_activity_by_trial[:, i, :], neural_activity_by_trial[:, j, :], axis=1)
		same_same_correlations.append(corr)
	for i, j in itertools.permutations(trial_indices_empty,  2):
		corr, _ = stats.pearsonr(neural_activity_by_trial[:, i, :], neural_activity_by_trial[:, j, :], axis=1)
		same_same_correlations.append(corr)

	same_cat_correlations = np.array(same_same_correlations)
	print(same_cat_correlations.shape)

	# do all pairwise combintations of trials across categories
	# find the correlation of activity by cell
	same_diff_correlations = []
	for i, j in itertools.product(trial_indices_active, trial_indices_empty):
		corr, _ = stats.pearsonr(neural_activity_by_trial[:, i, :], neural_activity_by_trial[:, j, :], axis=1)
		same_diff_correlations.append(corr)

	diff_cat_correlations = np.array(same_diff_correlations)
	print(diff_cat_correlations.shape)

	# plot as a violinplot and do statistical tests
	plt.violinplot([same_cat_correlations.flatten()[~np.isnan(same_cat_correlations.flatten())], diff_cat_correlations.flatten()[~np.isnan(diff_cat_correlations.flatten())]], showmeans=True)
	comparison = stats.mannwhitneyu(same_cat_correlations.flatten(), diff_cat_correlations.flatten(), alternative='greater', nan_policy='omit') # mann-whitney u for all trial combinations, all cells
	print(comparison)
	plt.title(f'p={comparison.pvalue:.4f}')
	plt.suptitle(f'all comparions, {mouseID}')
	plt.savefig(save_dir + 'single trial autocorrelations all comparisons.png')
	if show_figs:
		plt.show()
	else:
		plt.close()

	# find the average correlation across trials per cell
	cell_mean_within = np.nanmean(same_cat_correlations, axis=0)
	cell_mean_across = np.nanmean(diff_cat_correlations, axis=0)

	# plot as violin plot
	plt.violinplot([cell_mean_within, cell_mean_across], showmeans=True)

	# paired t-test by cells
	comparison = stats.ttest_rel(cell_mean_within, cell_mean_across, alternative='greater', nan_policy='omit') # paired t-test for per cell, averaged across trials
	print(comparison)
	print(np.nanmean(cell_mean_within)-np.nanmean(cell_mean_across))
	plt.title(f'p={comparison.pvalue:.4f}')
	plt.suptitle(f'averaged across trials by cell, {mouseID}')
	plt.savefig(save_dir + 'single trial autocorrelations averaged by cell.png')
	if show_figs:
		plt.show()
	else:
		plt.close()

	all_mice_within_cat.extend(cell_mean_within)
	all_mice_cross_cat.extend(cell_mean_across)

# violin plot and paired t-test for all cells (all mice)
all_mice_within_cat = np.array(all_mice_within_cat)
all_mice_cross_cat = np.array(all_mice_cross_cat)
plt.violinplot([all_mice_within_cat, all_mice_cross_cat], showmeans=True)
comparison = stats.ttest_rel(all_mice_within_cat, all_mice_cross_cat, alternative='greater', nan_policy='omit') # paired t-test for per cell, averaged across trials
print(comparison)
print(np.nanmean(all_mice_within_cat)-np.nanmean(all_mice_cross_cat))
plt.title(f'p={comparison.pvalue:.4f}')
plt.suptitle(f'averaged across trials by cell, all mice')
plt.savefig(multimouse_fig_dir + 'single trial autocorrelations averaged by cell.png')
plt.show()




