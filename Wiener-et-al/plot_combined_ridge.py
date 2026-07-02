'''
plot results for the combined shared unique ridge regression
'''
import numpy as np
import matplotlib.pyplot as plt
import src.utils
import seaborn as sns
from scipy import stats
from pathlib import Path
data_dir = Path.cwd() / 'data'

lag='2'
splitter='KFold'
n_repeats=10
n_folds=2
test_size=0.3
mice = ['mouse22', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
multimouse_fig_dir = data_dir + '/results/cross_mouse_results/'
which_measure = 'median'
normalized=True

model_features = ['paw\ncentroid x', 'paw centroid\ny', 'paw\ncentroid z', 'aperature', 'splay', 'orientation', 'active unique\ncoupling', 'empty unique\ncoupling', 'shared\ncoupling']
multimouse_separate_r2 = []
multimouse_combined_r2 = []

for mouseID in mice:
	drive = data_dir + '/neural'
	mouse_dir = drive + '/' + mouseID + '/'
	save_dir = mouse_dir + 'carry_analysis/GLM/'
	fig_save_dir = save_dir+'figures/'

	if normalized:
		if splitter=='KFold':
			active_r2 = np.load(save_dir + f'active_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			empty_r2 = np.load(save_dir + f'empty_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
	else:
		if splitter=='KFold':
			active_r2 = np.load(save_dir + f'active_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			empty_r2 = np.load(save_dir + f'empty_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
	if which_measure=='mean':	
		separate_r2 = np.mean(np.concatenate((active_r2, empty_r2), axis=1), axis=1)
	elif which_measure=='median':
		separate_r2 = np.median(np.concatenate((active_r2, empty_r2), axis=1), axis=1)
	assert len(separate_r2)==len(active_r2[:, 1]), 'averaged over wrong axes'
	
	if normalized:
		combined_r2 = np.load(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values_normalized_features.npy')
	else:
		combined_r2 = np.load(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values.npy')
	
	if which_measure=='mean':
		combined_cv_r2 = np.mean(combined_r2, axis=1)
	elif which_measure=='median':
		combined_cv_r2 = np.median(combined_r2, axis=1)
	assert len(combined_cv_r2)==len(separate_r2), 'different number of neurons somehow'
	multimouse_separate_r2.extend(separate_r2)
	multimouse_combined_r2.extend(combined_cv_r2)

	plt.scatter(separate_r2, combined_cv_r2, color='k', marker='.', alpha=0.25)
	plt.xlabel('condition-specific r2')
	plt.ylabel('condition-combined r2')
	plt.plot([-1, 1], [-1, 1], color='grey', ls='--')
	plt.xlim([-0.1, 1])
	plt.ylim([-.1, 1])
	plt.title(f"{mouseID}")
	if normalized:
		plt.savefig(fig_save_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance normalized.png')
	else:
		plt.savefig(fig_save_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance.png')
	plt.close()

	if normalized:
		betas = np.load(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_betas_normalized_features.npy')
	else:
		betas = np.load(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_betas.npy')
	
	mean_betas = np.mean(betas, axis=(0, 2))
	sem_betas = stats.sem(betas, axis=(0, 2))
	plt.bar(model_features, mean_betas, color='cornflowerblue')
	plt.errorbar(model_features, mean_betas, yerr=sem_betas, ecolor='navy', ls='none')
	plt.xlabel('feature')
	plt.ylabel('model coefficient')
	plt.title(f'{mouseID}')
	if normalized:
		plt.savefig(fig_save_dir + 'condition-combined ridge encoding betas normalized.png')
	else:
		plt.savefig(fig_save_dir + 'condition-combined ridge encoding betas.png')
	plt.close()

	a = np.concatenate([separate_r2[:, np.newaxis], combined_cv_r2[:, np.newaxis]], axis=1)
	sns.violinplot(a, color='cornflowerblue')
	plt.xticks([0, 1], ['condition-\nspecific', 'condition-\ncombined'])
	plt.ylabel('cross-validated r2')
	plt.axhline(0, ls='--', color='k')
	plt.ylim([-0.1, 1])
	plt.suptitle(f'{mouseID}')
	ttest = stats.ttest_rel(separate_r2, combined_cv_r2, alternative='greater')
	print('stat=', ttest.statistic)
	print('p=', ttest.pvalue)
	plt.title(f'p={ttest.pvalue:.5f}')
	if normalized:
		plt.savefig(fig_save_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance violin normalized.png')
	else:
		plt.savefig(fig_save_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance violin.png')
	plt.close()

multimouse_combined_r2 = np.array(multimouse_combined_r2)
multimouse_separate_r2 = np.array(multimouse_separate_r2)

plt.scatter(multimouse_separate_r2, multimouse_combined_r2, color='k', marker='.', alpha=0.1)
plt.xlabel('condition-specific r\u00b2')
plt.ylabel('condition-combined r\u00b2')
plt.plot([-1, 1], [-1, 1], color='grey', ls='--')
plt.xlim([-0.1, 1])
plt.ylim([-.1, 1])
plt.title('all cells, all mice')
if normalized:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance normalized no m25.png')
else:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance no m25.png')
plt.show()

xy = np.vstack([multimouse_separate_r2, multimouse_combined_r2])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
mmsep, mmcombo, z = multimouse_separate_r2[idx], multimouse_combined_r2[idx], z[idx]
best_fit = stats.linregress(mmsep, mmcombo)
print(best_fit)

fig, ax = plt.subplots()
ax.scatter(mmsep, mmcombo, c=z, marker='.', cmap='hot', rasterized=True)
ax.plot([-1, 1], [-1, 1], color='grey', ls='--')
ax.plot(np.unique(mmsep), np.unique(mmsep)*best_fit.slope+best_fit.intercept, color='indianred', lw=2)
ax.set_xlabel('condition-specific r\u00b2')
ax.set_ylabel('condition-combined r\u00b2')
ax.set_xlim([-0.1, 1])
ax.set_ylim([-.1, 1])
#ax.set_title('all cells, all mice')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_box_aspect(1)
if normalized:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance colored by density normalized no m25.pdf', dpi=550)
else:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge encoding {which_measure} cv performance colored by density no m25.pdf', dpi=550)
plt.show()

print(np.sum(np.logical_and(multimouse_separate_r2>multimouse_combined_r2, multimouse_separate_r2>0)), " separate greater than combined")
print(np.sum(np.logical_and(multimouse_separate_r2<multimouse_combined_r2, multimouse_combined_r2>0)), " combined greater than separate")

a = np.concatenate([multimouse_separate_r2[:, np.newaxis], multimouse_combined_r2[:, np.newaxis]], axis=1)
sns.violinplot(a, color='cornflowerblue')
plt.xticks([0, 1], ['condition-\nspecific', 'condition-\ncombined'])
plt.ylabel('cross-validated r\u00b2')
plt.axhline(0, ls='--', color='k')
plt.ylim([-0.1, 1])
plt.suptitle('all cells, all mice')
ttest = stats.ttest_rel(multimouse_separate_r2, multimouse_combined_r2, alternative='greater')
print('stat=', ttest.statistic)
print('p=', ttest.pvalue)
plt.title(f'p={ttest.pvalue:.5f}')
if normalized:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge {which_measure} cv encoding performance violin normalized no m25.png')
else:
	plt.savefig(multimouse_fig_dir + f'condition-specific vs condition-combined ridge {which_measure} cv encoding performance violin no m25.png')
plt.show()