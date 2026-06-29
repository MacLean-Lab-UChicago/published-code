'''
compare PETHs quantitatively for carries
'''
import numpy as np
import src.IO
import src.utils
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
data_dir = Path.cwd() / 'data'

cross_mouse_fig_dir = data_dir + '/results/cross_mouse_results/'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
t_pre = 4 # frames
t_post = 4 # frames
time_in_trial=t_pre+t_post
bootstrap_count = 1000
carry_classes = [1, 2] # active, empty

cross_mouse_correlations = [[], [], []]
for mouseID in mice:
	days = src.IO.get_carry_days(mouseID)
	drive = src.IO.get_drive(mouseID)
	mouse_dir = data_dir + '/neural/' + mouseID + '/'
	fig_save_dir = data_dir + 'results/single_mouse_results/' + mouseID + '/figures/'
	
	reg_inds = src.utils.load_registered_cells(mouse_dir, days)

	F_multi = [[] for i in range(len(days))]
	carry_time_multi = [[[] for i in range(len(days))] for i in range(len(carry_classes))]
	carry_labels_multi = [[] for i in range(len(days))]
	for i, day in enumerate(days):
		# load in data
		s2p_fld = src.IO.get_s2p_fld(mouseID, day)
		f = np.load(s2p_fld + 'cascade_spks.npy')
		F_reg = f[reg_inds[i], :]
		n_neurons = F_reg.shape[0]
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		F_multi[i] = F_reg
		for c_class, cat in enumerate(carry_classes):
			carry_time_multi[c_class][i] = carry_times[carry_labels==cat]
	
	# remove cells with NaNs on any day
	NaN_indices = np.zeros(n_neurons, dtype=bool)
	for day_i, F in enumerate(F_multi):
		# plt.imshow(F)
		# plt.show()
		NaN = np.isnan(F).all(axis=1)
		NaN_indices = np.logical_or(NaN_indices, NaN)
	print(f'{np.sum(NaN_indices)} cells are NaN, removing...')
	n_neurons = np.sum(~NaN_indices)

	for day_i, F in enumerate(F_multi):
		F_multi[day_i] = F[~NaN_indices, :]

	# calculate PETHs
	F_wet, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_time_multi[0], t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)
	F_dry, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_time_multi[1], t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)

	# do random shuffles
	wet_shuffled = np.matrix.copy(F_wet)
	np.random.shuffle(wet_shuffled)
	dry_shuffled = np.matrix.copy(F_dry)
	np.random.shuffle(dry_shuffled)

	# get correlations across categories and compared to shuffle controls
	correlations = []
	cross_cat_corr, cross_cat_pval = stats.pearsonr(F_wet, F_dry, axis=1)
	shuff_corr, shuff_pval = stats.pearsonr(wet_shuffled, F_dry, axis=1)

	# repeat shuffles and find correlations
	repeated_shuffle_corr = []
	for boot in range(bootstrap_count):
		wet_shuffled = np.matrix.copy(F_wet)
		np.random.shuffle(wet_shuffled)
		dry_shuffled = np.matrix.copy(F_dry)
		np.random.shuffle(dry_shuffled)
		shuff_corr, shuff_pval = stats.pearsonr(wet_shuffled, F_dry, axis=1)
		repeated_shuffle_corr.extend(shuff_corr)
		shuff_corr, shuff_pval = stats.pearsonr(F_wet, dry_shuffled, axis=1)
		repeated_shuffle_corr.extend(shuff_corr)

	# within category bootstrap
	wi_cat_corr_bootstrapped = []
	
	for idx in range(len(carry_classes)):
		# convert multiday into something that can be indexed by a random choice
		times = carry_time_multi[idx]
		cross_day_times = []
		for day_i in range(len(days)):
			cross_day_times.extend([f'{day_i}_{int(time)}' for time in times[day_i]])
		#print(cross_day_times)
		for repeat in range(bootstrap_count):
			# randomly split the data in half
			split = np.random.choice(len(cross_day_times), size=int(.5*len(cross_day_times)), replace=False)
			# restore the multiday nested list from the split indices
			carry_times_1half = [[] for i in range(len(days))]
			carry_times_2half = [[] for i in range(len(days))]
			for time in cross_day_times:
				day = int(time.split('_')[0])
				t_stamp = int(time.split('_')[1])
				if time in np.array(cross_day_times, dtype=object)[split]:
					carry_times_1half[day].append(t_stamp)
				else:
					carry_times_2half[day].append(t_stamp)
			for day_i in range(len(days)):
				carry_times_1half[day_i] = np.array(carry_times_1half[day_i])
				carry_times_2half[day_i] = np.array(carry_times_2half[day_i])

			# calculate PETHs			
			half1_PETH, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_times_1half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)
			half2_PETH, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_times_2half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)

			# find correlations and p-values
			corr, pval = stats.pearsonr(half1_PETH, half2_PETH, axis=1)
			wi_cat_corr_bootstrapped.append(corr)

	cross_cat_corr_bootstrapped = []
	# convert multiday into something that can be indexed by a random choice
	times_0 = carry_time_multi[0]
	times_1 = carry_time_multi[1]
	cross_day_times_0 = []
	cross_day_times_1 = []
	for day_i in range(len(days)):
		cross_day_times_0.extend([f'{day_i}_{int(time)}' for time in times_0[day_i]])
		cross_day_times_1.extend([f'{day_i}_{int(time)}' for time in times_1[day_i]])

	#print(cross_day_times)
	for repeat in range(bootstrap_count):
		# randomly split the data in half
		split_0 = np.random.choice(len(cross_day_times_0), size=int(.5*len(cross_day_times_0)), replace=False)
		split_1 = np.random.choice(len(cross_day_times_1), size=int(.5*len(cross_day_times_1)), replace=False)
		# restore the multiday nested list from the split indices
		carry_times_1half = [[] for i in range(len(days))]
		carry_times_2half = [[] for i in range(len(days))]
		for time in cross_day_times_0:
			day = int(time.split('_')[0])
			t_stamp = int(time.split('_')[1])
			if time in np.array(cross_day_times_0, dtype=object)[split_0]:
				carry_times_1half[day].append(t_stamp)
		for time in cross_day_times_1:
			day = int(time.split('_')[0])
			t_stamp = int(time.split('_')[1])
			if time in np.array(cross_day_times_1, dtype=object)[split_1]:
				carry_times_2half[day].append(t_stamp)
		
		for day_i in range(len(days)):
			carry_times_1half[day_i] = np.array(carry_times_1half[day_i])
			carry_times_2half[day_i] = np.array(carry_times_2half[day_i])
		# calculate PETHs
		half1_PETH, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_times_1half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)
		half2_PETH, ind = src.utils.plot_PETH_multiday(days, F_multi, carry_times_2half, t_pre=t_pre, t_post=t_post, sort=False, sort_ind=None, norm=False)

		# find correlations and p-values
		corr, pval = stats.pearsonr(half1_PETH, half2_PETH, axis=1)
		cross_cat_corr_bootstrapped.append(corr)

	# average correlations by cell - necessary for paired t-test
	# z-transform the correlations before averaging to preserve normality in a correlation space
	wi_cat_corr = np.tanh(np.nanmean(np.arctanh(wi_cat_corr_bootstrapped), axis=0)) 
	cross_cat_corr_boot = np.tanh(np.nanmean(np.arctanh(cross_cat_corr_bootstrapped), axis=0))
	shuff_corr = np.array(repeated_shuffle_corr)
	correlations.append(wi_cat_corr)
	correlations.append(cross_cat_corr_boot)
	correlations.append(shuff_corr)

	cross_mouse_correlations[0].extend(wi_cat_corr)
	cross_mouse_correlations[1].extend(cross_cat_corr_boot)
	cross_mouse_correlations[2].extend(shuff_corr)

	# plot correlations as a violin plot
	colors = ['#875A65', '#C28884', '#E2B98D']  # Different colors for each violin 
	labels = ['within-category', 'cross-category', 'CellID Shuffle']
	positions = np.arange(1, len(correlations) + 1)
	fig, ax = plt.subplots(figsize=(4.15, 3))
	parts = ax.violinplot(correlations, positions, showmeans=True, vert=True)
	for i, pc in enumerate(parts['bodies']):
		pc.set_facecolor(colors[i])
		pc.set_edgecolor(colors[i])
		pc.set_alpha(0.7)

	# Set colors for other elements
	for part in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
		parts[part].set_color('black')
	xmin, xmax = ax.get_xlim()
	#ax.hlines(percentile_95, xmin, xmax, colors = 'red', linestyles = 'dashed')
	ax.set_xticks(positions)
	ax.set_xticklabels(labels)

	# Perform pairwise t-tests/Mann-Whitney U tests and add significance lines
	bottom, top = ax.get_ylim()
	y_range = top - bottom
	y_max = max([np.nanmax(d) for d in correlations]) + .15
	y_start = y_max + .05*y_max  # Initial y position for the first line
	y_step = .075*2  # Spacing between lines

	# Iterate over group combinations
	for i, (g1, g2) in enumerate([[0, 1], [1, 2]]):
		if g2==2:
			comparison = stats.mannwhitneyu(correlations[g1], correlations[g2], alternative='greater') # when comparing to shuffle use Mann-Whitney U
		else:
			comparison = stats.ttest_rel(correlations[g1], correlations[g2], alternative='greater') # when comparing to within-sample use paired t-test
		
		# plot the significance lines, increasing in height
		bar_height = (y_range * 0.15 * i) + top
		bar_tips = bar_height - (y_range * 0.02)
		ax.plot(
			[positions[g1], positions[g1], positions[g2], positions[g2]],
			[bar_tips, bar_height, bar_height, bar_tips], lw=1.5, c='k')
		# Add p-value annotation
		if comparison.pvalue < .001:
			p_text = '***'
			text_height = bar_height - (y_range * 0.01)
		elif comparison.pvalue < .01:
			p_text = '**'
			text_height = bar_height - (y_range * 0.01)
		elif comparison.pvalue < .05:
			p_text = '*'
			text_height = bar_height - (y_range * 0.01)
		else:
			p_text = 'n.s.'
			text_height = bar_height + (y_range * 0.005)
		if i==0:
			p = comparison.pvalue
		ax.text((positions[g1] + positions[g2]) * 0.5, text_height, p_text, ha='center', va='bottom', c='k', fontsize=10)
		print(f'{i}: p={comparison.pvalue}')
	ax.set_ylabel('Correlation', fontsize=11)
	ax.axhline(y=0, ls='--', color='#A5A5A5')
	#ax.set_title('Correlation Between Population Activity Across Strategies')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.tight_layout()
	plt.savefig(fig_save_dir+'wi vs x-category PETH correlations.png', dpi=660)
	print(f'p = {p:.3f}, effect size = {np.nanmean(cross_mouse_correlations[0])-np.nanmean(cross_mouse_correlations[1])}')
	plt.show()

# plot across all cells (all mice)
colors = ['#875A65', '#C28884', '#E2B98D']  # Different colors for each violin 
labels = ['within-category', 'cross-category', 'CellID Shuffle']
positions = np.arange(1, len(cross_mouse_correlations) + 1)
fig, ax = plt.subplots(figsize=(4.15, 3))
parts = ax.violinplot(cross_mouse_correlations, positions, showmeans=True, vert=True)
for i, pc in enumerate(parts['bodies']):
	pc.set_facecolor(colors[i])
	pc.set_edgecolor(colors[i])
	pc.set_alpha(0.7)

# Set colors for other elements
for part in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
	parts[part].set_color('black')
xmin, xmax = ax.get_xlim()
#ax.hlines(percentile_95, xmin, xmax, colors = 'red', linestyles = 'dashed')
ax.set_xticks(positions)
ax.set_xticklabels(labels)

# Perform pairwise t-tests/Mann-Whitney U tests and add significance lines
bottom, top = ax.get_ylim()
y_range = top - bottom
y_max = max([np.nanmax(d) for d in cross_mouse_correlations]) + .15
y_start = y_max + .05*y_max  # Initial y position for the first line
y_step = .075*2  # Spacing between lines

# Iterate over group combinations
for i, (g1, g2) in enumerate([[0, 1], [1, 2]]):
	if g2==2:
		comparison = stats.mannwhitneyu(cross_mouse_correlations[g1], cross_mouse_correlations[g2], alternative='greater') # when comparing to shuffle use Mann-Whitney U
	else:
		comparison = stats.ttest_rel(cross_mouse_correlations[g1], cross_mouse_correlations[g2], alternative='two-sided') # when comparing to within-sample use paired t-test
	
	# plot the significance lines, increasing in height
	bar_height = (y_range * 0.15 * i) + top
	bar_tips = bar_height - (y_range * 0.02)
	ax.plot(
		[positions[g1], positions[g1], positions[g2], positions[g2]],
		[bar_tips, bar_height, bar_height, bar_tips], lw=1.5, c='k')
	# Add p-value annotation
	if comparison.pvalue < .001:
		p_text = '***'
		text_height = bar_height - (y_range * 0.01)
	elif comparison.pvalue < .01:
		p_text = '**'
		text_height = bar_height - (y_range * 0.01)
	elif comparison.pvalue < .05:
		p_text = '*'
		text_height = bar_height - (y_range * 0.01)
	else:
		p_text = 'n.s.'
		text_height = bar_height + (y_range * 0.005)
	ax.text((positions[g1] + positions[g2]) * 0.5, text_height, p_text, ha='center', va='bottom', c='k', fontsize=10)
	if i==0:
		p = comparison.pvalue
	print(f'{i}: p={comparison.pvalue}')
ax.set_ylabel('Correlation', fontsize=11)
ax.axhline(y=0, ls='--', color='#A5A5A5')
#ax.set_title('Correlation Between Population Activity Across Strategies')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
print(f'p = {p:.3f}, effect size = {np.nanmean(cross_mouse_correlations[0])-np.nanmean(cross_mouse_correlations[1]):.3f}')
plt.savefig(cross_mouse_fig_dir+'wi vs x-category PETH half correlations.png', dpi=660)
plt.show()

