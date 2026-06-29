'''
it got way too confusing to have all the plotting on the same script as the models
so here is plotting results for the encoding models
this code does not plot the results applied to drops, that is in the drop code
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import src.utils

lag='2'
splitter='KFold'
n_repeats=10
n_folds=2
test_size=0.3
normalized=True

mice = ['mouse22', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/encoding model figures/'
which_measure = 'median'

full_model=True
plot_LOO_models=True
include_sign_flip=True
kin_vs_full=True
magnitude_LOO = True
permutations = True
plot_LOO_scatter_by_magnitude = True

show_plots=True

if kin_vs_full:
	# compare kin only to kinematics-and-coupling full model
	full_dict = {'mouse':[], 'r2':[], 'model':[]}
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		for category in ['active', 'empty']:
			if normalized:
				kin_r2 = np.load(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
				r2_values = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				kin_r2 = np.load(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
				r2_values = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			if which_measure=='median':
				med_r2 = np.median(r2_values, axis=1) # median over CV folds
				kin_med_r2 = np.median(kin_r2, axis=1)
			elif which_measure=='mean':
				med_r2 = np.mean(r2_values, axis=1) # mean over CV folds
				kin_med_r2 = np.mean(kin_r2, axis=1)

			full_dict['r2'].extend(med_r2)
			full_dict['r2'].extend(kin_med_r2)
			full_dict['model'].extend(np.ones(len(med_r2), dtype=object)*'kinematic-coupling')
			full_dict['model'].extend(np.ones(len(med_r2), dtype=object)*'kinematic only')
			full_dict['mouse'].extend(np.ones(len(med_r2)*2, dtype=object)*mouseID)
			

	# combined across mice
	# plot as a custom color matplotlib plot (thanks Claude)
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(full_dict)

	# assign colors
	categories = df['model'].unique()

	fig, ax = plt.subplots()

	# Build list of data arrays, one per category (in order)
	data_by_cat = [df.loc[df['model'] == cat, 'r2'].values for cat in categories]

	# Plot violins
	parts = ax.violinplot(data_by_cat, positions=[0, 1], showmedians=True)

	for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
		if partname in parts:
			parts[partname].set_edgecolor('black')
			parts[partname].set_linewidth(1.5)

	ax.set_xticks([0, 1])
	ax.set_xticklabels(['kinematics-coupling', 'kinematics-only'])
	ax.set_xlabel('model')
	ax.set_ylabel('r\u00b2')
	ax.axhline(0, ls='--', color='k')
	ax.set_ylim(-.1, .8)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# paired t-test to compared cv r2 from kinematics to kinematics and coupling
	result = stats.ttest_rel(np.array(full_dict['r2'])[np.array(full_dict['model'], dtype=object)=='kinematic-coupling'], np.array(full_dict['r2'])[np.array(full_dict['model'], dtype=object)=='kinematic only'], alternative='greater')
	print(result)
	ax.plot([0, 1], np.ones(2)*(np.max(data_by_cat[0])+.025), color='k')
	if result.pvalue<0.001:
		ax.text(.5, np.max(data_by_cat[0])+0.05, '***', horizontalalignment='center')
	elif result.pvalue<0.01:
		ax.text(.5, np.max(data_by_cat[0])+0.05, '**', horizontalalignment='center')
	elif result.pvalue<0.05:
		ax.text(.5, np.max(data_by_cat[0])+0.05, '*', horizontalalignment='center')
	else:
		ax.text(.5, np.max(data_by_cat[0])+0.05, 'n.s.', horizontalalignment='center')
	

	plt.tight_layout()
	plt.savefig(multimouse_fig_dir + f'kinematic vs kinematic-coupling model performance {which_measure} across folds.pdf', dpi=660)
	if show_plots:
		plt.show()
	else:
		plt.close()


if magnitude_LOO:
	LOO_models = ['strong', 'medium', 'weak', 'zero']
	removed_edge_proportions = [.01, .05, .1]
	LOO_dr2_dict = {'mouse':[], 'delta r2':[], 'left out':[]}
	ctrl_LOO_dr2_dict = {'mouse':[], 'delta r2':[], 'left out':[], 'proportion removed': []}
	by_mouse_LOO_dict = {mouseID:{} for mouseID in mice}
	dim_matched_by_mouse_LOO_dict = {mouseID:{} for mouseID in mice}
	for mouseID in mice:
		# load in data
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		for cat in ['active', 'empty']:
			if normalized:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges_NORMALIZED.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_NORMALIZED.npy')
				r2_prop_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_by_proportion_of_edges_{removed_edge_proportions}_NORMALIZED.npy')
			else:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength.npy')
				r2_prop_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_by_proportion_of_edges_{removed_edge_proportions}.npy')
			well_modelled_idx = np.logical_or(np.mean(r2_full_model, axis=1)>0.05, np.median(r2_full_model, axis=1)>0.05)
			print(np.sum(well_modelled_idx))
			print(np.sum(np.isnan(r2_prop_control[well_modelled_idx, 0, 0, 0])))
			# take the mean across folds to get cv delta r2
			for model_i, model in enumerate(LOO_models):
				LOO_dr2_dict['delta r2'].extend(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1))
				LOO_dr2_dict['mouse'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				LOO_dr2_dict['left out'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*model)
			for prop_i, prop in enumerate(removed_edge_proportions):
				r2_dim_control = r2_prop_control[:, :, prop_i, :]
				for model_i, model in enumerate(LOO_models[:-1]):
					ctrl_LOO_dr2_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1))
					ctrl_LOO_dr2_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
					ctrl_LOO_dr2_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*model)
					ctrl_LOO_dr2_dict['proportion removed'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*prop)


		# plot LOO performance for each mouse
		by_mouse_LOO_dict[mouseID] = {'delta r2':np.array(LOO_dr2_dict['delta r2'])[np.array(LOO_dr2_dict['mouse'], dtype=object)==mouseID], 'left out':np.array(LOO_dr2_dict['left out'])[np.array(LOO_dr2_dict['mouse'], dtype=object)==mouseID]}
		sns.violinplot(data=by_mouse_LOO_dict[mouseID], x='left out', y='delta r2', cut=0)
		plt.ylim(-1.01, .01)
		plt.title(f'{mouseID} weight magnitude LOO delta r2')
		if show_plots:
			plt.show()
		else:
			plt.close()

		# plot FC count-matched LOO performance for each mouse
		dim_matched_by_mouse_LOO_dict[mouseID] = {'delta r2':np.array(ctrl_LOO_dr2_dict['delta r2'])[np.array(ctrl_LOO_dr2_dict['mouse'], dtype=object)==mouseID], 'left out':np.array(ctrl_LOO_dr2_dict['left out'])[np.array(ctrl_LOO_dr2_dict['mouse'], dtype=object)==mouseID]}
		sns.violinplot(data=dim_matched_by_mouse_LOO_dict[mouseID], x='left out', y='delta r2', cut=0)
		plt.ylim(-.2, .025)
		plt.title(f'{mouseID} count-matched weight magnitude LOO delta r2')
		if show_plots:
			plt.show()
		else:
			plt.close()

	# plot LOO performance combined across mice
	sns.violinplot(data=LOO_dr2_dict, x='left out', y='delta r2', cut=0)
	plt.title('Delta R2 Compared to Full Model')
	plt.axhline(0, ls='--', color='k')
	plt.ylim(-.4, .1)
	#plt.savefig(multimouse_fig_dir + 'weight magnitude edges delta r2s violin.png')
	if show_plots:
		plt.show()
	else:
		plt.close()
	
	
	# plot count matched LOO as a custom color matplotlib plot (thanks Claude)
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(ctrl_LOO_dr2_dict)

	# assign colors
	categories = df['left out'].unique()

	fig, ax = plt.subplots()

	# Build list of data arrays, one per category (in order)
	data_by_cat = [df.loc[df['left out'] == cat, 'delta r2'].values for cat in categories]

	# Plot violins
	parts = ax.violinplot(data_by_cat, positions=[0, 1, 2], showmedians=True)

	for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
		if partname in parts:
			parts[partname].set_edgecolor('black')
			parts[partname].set_linewidth(1.5)

	ax.set_xticks([0, 1, 2])
	ax.set_xticklabels(['strong', 'medium', 'weak'])
	ax.set_xlabel('FC-count matched left out')
	ax.set_ylabel('delta r\u00b2')
	ax.axhline(0, ls='--', color='k')
	
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# paired t-test to compare leaving out different count-matched FC by magnitude
	strong_vs_medium = stats.ttest_rel(data_by_cat[0], data_by_cat[1], alternative='less', )
	strong_vs_weak = stats.ttest_rel(data_by_cat[0], data_by_cat[2], alternative='less', )
	medium_vs_weak = stats.ttest_rel(data_by_cat[1], data_by_cat[2], alternative='less', )
	print(strong_vs_medium)
	print(strong_vs_weak)
	print(medium_vs_weak)

	# plot significance lines
	ax.plot([0,1], np.ones(2)*(np.min(data_by_cat[0])-.005), color='k')
	if strong_vs_medium.pvalue<0.001:
		ax.text(0.5, np.min(data_by_cat[0])-0.005, '***', horizontalalignment='center')
	elif strong_vs_medium.pvalue<0.01:
		ax.text(0.5, np.min(data_by_cat[0])-0.005, '**', horizontalalignment='center')
	elif strong_vs_medium.pvalue<0.05:
		ax.text(0.5, np.min(data_by_cat[0])-0.005, '*', horizontalalignment='center')
	else:
		ax.text(0.5, np.min(data_by_cat[0])-0.005, 'n.s.', horizontalalignment='center')

	ax.plot([1, 2], np.ones(2)*(np.min(data_by_cat[1])-.005), color='k')
	if medium_vs_weak.pvalue<0.001:
		ax.text(1.5, np.min(data_by_cat[1])-0.005, '***', horizontalalignment='center')
	elif medium_vs_weak.pvalue<0.01:
		ax.text(1.5, np.min(data_by_cat[1])-0.005, '**', horizontalalignment='center')
	elif medium_vs_weak.pvalue<0.05:
		ax.text(1.5, np.min(data_by_cat[1])-0.005, '*', horizontalalignment='center')
	else:
		ax.text(1.5, np.min(data_by_cat[1])-0.005, 'n.s.', horizontalalignment='center')

	ax.plot([0,2], np.ones(2)*(np.min(data_by_cat[0])-0.015), color='k')
	if strong_vs_weak.pvalue<0.001:
		ax.text(1, np.min(data_by_cat[0])-0.015, '***', horizontalalignment='center')
	elif strong_vs_weak.pvalue<0.01:
		ax.text(1, np.min(data_by_cat[0])-0.015, '**', horizontalalignment='center')
	elif strong_vs_weak.pvalue<0.05:
		ax.text(1, np.min(data_by_cat[0])-0.015, '*', horizontalalignment='center')
	else:
		ax.text(1, np.min(data_by_cat[0])-0.015, 'n.s.', horizontalalignment='center')


	ax.set_ylim(-.17, .02)
	plt.tight_layout()
	plt.savefig(multimouse_fig_dir + f'weight magnitude edges delta r2s violin dim matched.pdf', dpi=660)
	if show_plots:
		plt.show()
	else:
		plt.close()


if full_model:
	# plot violin plot for same vs opposite category weights
	r2_dict={'mouse':[], 'r2':[], 'couplings':[]}
	all_folds_dict={'mouse':[], 'r2':[], 'couplings':[]}
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		for category in ['active', 'empty']:
			if normalized:
				r2_values = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
				opp_cat_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				r2_values = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
				opp_cat_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			print(f'{mouseID}, {category}, {np.median(np.mean(r2_values, axis=1))}')

			all_folds_dict['r2'].extend(r2_values.flatten())
			all_folds_dict['mouse'].extend(np.ones(len(r2_values.flatten()), dtype=object)*mouseID)
			all_folds_dict['couplings'].extend(np.ones(len(r2_values.flatten()), dtype=object)*'within category weights')
			if which_measure=='median':
				med_r2 = np.median(r2_values, axis=1) # median over CV folds
			elif which_measure=='mean':
				med_r2 = np.mean(r2_values, axis=1) # mean over CV folds
			print(np.median(med_r2))
			r2_dict['r2'].extend(med_r2)
			r2_dict['mouse'].extend(np.ones(len(med_r2), dtype=object)*mouseID)
			r2_dict['couplings'].extend(np.ones(len(med_r2), dtype=object)*'within category weights')
			all_folds_dict['r2'].extend(opp_cat_r2.flatten())
			all_folds_dict['mouse'].extend(np.ones(len(opp_cat_r2.flatten()), dtype=object)*mouseID)
			all_folds_dict['couplings'].extend(np.ones(len(opp_cat_r2.flatten()), dtype=object)*'opposite category weights')
			if which_measure=='median':
				med_opp_r2 = np.median(opp_cat_r2, axis=1) # median over CV folds
			elif which_measure=='mean':
				med_opp_r2 = np.mean(opp_cat_r2, axis=1) # mean over CV folds
			r2_dict['r2'].extend(med_opp_r2)
			r2_dict['mouse'].extend(np.ones(len(med_r2), dtype=object)*mouseID)
			r2_dict['couplings'].extend(np.ones(len(med_r2), dtype=object)*'opposite category weights')


	print(np.max(r2_dict['r2']))
	print(len(r2_dict['r2']))
	# plot per mouse cross-validated r2
	sns.violinplot(data=r2_dict, x='mouse', y='r2', hue='couplings', split=True, inner='quart', gap=0.06, cut=0)
	plt.ylim(-.05, .7)
	plt.ylabel('cross-validated r\u00b2')
	plt.title('Ridge Linear Regression Performance with Couplings By Mouse')
	if show_plots:
		plt.show()
	else:
		plt.close()

	print(np.max(all_folds_dict['r2']))
	print(len(all_folds_dict['r2']))
	# plot per mouse r2 across all folds
	sns.violinplot(data=all_folds_dict, x='mouse', y='r2', hue='couplings', split=True, inner='quart', gap=0.06, cut=0)
	plt.ylim(-.05, 1)
	plt.ylabel('r\u00b2')
	plt.title('Ridge Linear Regression Performance with Couplings By Mouse, All Folds')
	if show_plots:
		plt.show()
	else:
		plt.close()


	# combined across mice
	sns.violinplot(data=r2_dict, x='couplings', y='r2', cut=0)
	result = stats.ttest_ind(np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='within category weights'], np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='opposite category weights'], alternative='greater')
	plt.ylim(-.05, 1)
	plt.ylabel('cross-validated r\u00b2')
	plt.suptitle('Ridge Linear Regression Performance with Couplings')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_plots:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')


	# combined across mice, all folds
	sns.violinplot(data=all_folds_dict, x='couplings', y='r2', cut=0)
	result = stats.ttest_ind(np.array(all_folds_dict['r2'])[np.array(all_folds_dict['couplings'], dtype=object)=='within category weights'], np.array(all_folds_dict['r2'])[np.array(all_folds_dict['couplings'], dtype=object)=='opposite category weights'], alternative='greater')
	plt.ylim(-.05, 1)
	plt.ylabel('r\u00b2')
	plt.suptitle('Ridge Linear Regression Performance with Couplings, All Folds')
	plt.title(f't-test, p={result.pvalue:.4f}')
	if show_plots:
		plt.show()
	else:
		plt.close()
	print(f'p={result.pvalue}')


	# scatterplot per mouse
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		plt.scatter(np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['couplings'], dtype=object)=='opposite category weights')], 
			np.array(r2_dict['r2'])[np.logical_and(np.array(r2_dict['mouse'], dtype=object)==mouseID, np.array(r2_dict['couplings'], dtype=object)=='within category weights')], 
			color='k', marker='.', alpha=0.3)
		plt.xlabel('opposite category r\u00b2')
		plt.ylabel('within category r\u00b2')
		plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
		plt.title(f'{mouseID}')
		plt.xlim(-.05, 1)
		plt.ylim(-.05, 1)
		if normalized:
			if splitter=='Shuffle':
				plt.savefig(fig_save_dir + f'within vs opposite {which_measure} r2 scatterplot trial split test {test_size} NORMALIZED.png')
			elif splitter=='KFold':
				plt.savefig(fig_save_dir + f'within vs opposite {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split NORMALIZED.png')
		else:	
			if splitter=='Shuffle':
				plt.savefig(fig_save_dir + f'within vs opposite {which_measure} r2 scatterplot trial split test {test_size}.png')
			elif splitter=='KFold':
				plt.savefig(fig_save_dir + f'within vs opposite {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

	# scatterplot combined across mice
	plt.scatter(np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='opposite category weights'], 
		np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='within category weights'], 
			color='k', marker='.', alpha=0.15)
	plt.xlabel('opposite category r\u00b2')
	plt.ylabel('within category r\u00b2')
	plt.plot([-.05, 1], [-.05, 1], '--', color='grey')
	plt.xlim(-.05, 1)
	plt.ylim(-.05, 1)
	plt.title('Encoding Performance, Within- Vs Opposite-Cateogory Coupldings')
	if normalized:
		if splitter=='Shuffle':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot trial split test {test_size} NORMALIZED no m25.png')
		if splitter=='KFold':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split NORMALIZED no m25.png')
	else:
		if splitter=='Shuffle':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot trial split test {test_size} no m25.png')
		if splitter=='KFold':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split no m25.png')
	if show_plots:
		plt.show()
	else:
		plt.close()

	# scatterplot combined across mice but with color corresponding to density
	# credit to Harold Rockwell
	# restricts the gaussian kde only with the points that will actually be in frame
	from scipy.stats import gaussian_kde
	xy = np.vstack([np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='opposite category weights'], np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='within category weights']])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	opp_cat, wi_cat, z = np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='opposite category weights'][idx], np.array(r2_dict['r2'])[np.array(r2_dict['couplings'], dtype=object)=='within category weights'][idx], z[idx]

	# calculate line of best fit
	best_fit = stats.linregress(wi_cat, opp_cat)
	print(best_fit)
	r = stats.pearsonr(opp_cat, wi_cat)
	print(r)

	fig, ax = plt.subplots()
	ax.scatter(opp_cat, wi_cat, c=z, marker='.', cmap='hot', rasterized=True)
	ax.plot([-.05, 1], [-.05, 1], '--', color='grey')
	ax.plot(best_fit.slope*np.unique(wi_cat) + best_fit.intercept, np.unique(wi_cat), color='indianred', lw=2)
	ax.set_xlabel('mismatched couplings r\u00b2')
	ax.set_ylabel('matched couplings r\u00b2')
	ax.set_xlim(-.05, 1)
	ax.set_ylim(-.05, 1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_box_aspect(1)

	if normalized:
		if splitter=='Shuffle':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot colored by density trial split test {test_size} NORMALIZED no m25.png')
		if splitter=='KFold':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot colored by density trial {n_repeats}x{n_folds}Fold split NORMALIZED no m25.pdf', dpi=500)
	else:
		if splitter=='Shuffle':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot colored by density trial split test {test_size} no m25.png')
		if splitter=='KFold':
			plt.savefig(multimouse_fig_dir + f'within vs opposite category couplings {which_measure} r2 scatterplot colored by density trial {n_repeats}x{n_folds}Fold split no m25.png')
	if show_plots:
		plt.show()
	else:
		plt.close()

	# plot scatterplot colored by trial category 
	# (check that there is not a large separation of performance by category)
	fig, ax = plt.subplots()
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		colors = ['darkred', 'navy']
		labels=['active grasp', 'empty grasp']
		for i, cat in enumerate(['active', 'empty']):
			if normalized:
				if splitter=='Shuffle':
					r2_values = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
					opp_cat_r2 = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
				elif splitter=='KFold':
					r2_values = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
					opp_cat_r2 = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				if splitter=='Shuffle':
					r2_values = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat.npy')
					opp_cat_r2 = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat.npy')
				elif splitter=='KFold':
					r2_values = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
					opp_cat_r2 = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			if which_measure=='mean':
				avg_r2 = avged_r2_values = np.mean(r2_values, axis=1) # mean over CV folds
			elif which_measure=='median':
				avg_r2 = avged_r2_values = np.median(r2_values, axis=1) # median over CV folds
			if which_measure=='mean':
				avg_opp_r2 = np.mean(opp_cat_r2, axis=1) # mean over CV folds
			elif which_measure=='median':
				avg_opp_r2 = np.median(opp_cat_r2, axis=1) # median over CV folds
			if mouseID==mice[-1]:
				ax.scatter(avg_opp_r2, avg_r2, color=colors[i], marker='.', alpha=0.15, label=labels[i])
			else:
				ax.scatter(avg_opp_r2, avg_r2, color=colors[i], marker='.', alpha=0.15)

	ax.set_ylabel('within-category CV r\u00b2')
	ax.set_xlabel('across-category CV r\u00b2')
	ax.plot([-1, 1], [-1, 1], ls='--', color='grey')
	ax.legend()
	ax.set_title('Encoding Performance, Within- Vs Opposite-Cateogory Coupldings')
	ax.set_xlim(-.1, 1)
	ax.set_ylim(-.1, 1)
	if show_plots:
		plt.show()
	else:
		plt.close()

if plot_LOO_models:
	# plot unique/shared LOO delta r2
	LOO_models = ['category_unique_edges', 'other_cat_unique_edges', 'shared_edges', 'sign_flip']
	dim_control_models = ['undersamp_cat_unique_to_shared', 'undersamp_other_cat_unique_to_shared', 'undersamp_cat_unique_to_sign_flip', 
		'undersamp_other_cat_unique_to_sign_flip', 'undersamp_shared_to_sign_flip', 'undersamp_larger_cat_to_other_cat']
	LOO_dr2_dict = {'mouse':[], 'delta r2':[], 'left out':[]}
	ctrl_LOO_dr2_dict = {'mouse':[], 'delta r2':[], 'left out':[]}
	perc_LOO_dr2_dict = {'mouse':[], 'percent delta r2':[], 'left out':[]}
	perc_ctrl_LOO_dr2_dict = {'mouse':[], 'percent delta r2':[], 'left out':[]}
	by_mouse_LOO_dict = {mouseID:{} for mouseID in mice}
	dim_matched_by_mouse_LOO_dict = {mouseID:{} for mouseID in mice}
	same_size_edges_dict = {'mouse':[], 'delta r2':[], 'left out':[]}
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		for cat in ['active', 'empty']:
			if normalized:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges_NORMALIZED.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_NORMALIZED.npy')
				r2_dim_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control_NORMALIZED.npy')
			else:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness.npy')
				r2_dim_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control.npy')
			dr2_values = np.zeros((r2_full_model.shape[0], len(LOO_models)))*np.nan
			ctrl_dr2_values = np.zeros((r2_full_model.shape[0], len(dim_control_models)))*np.nan
			well_modelled_idx = ~np.isnan(r2_LOO[:, 0, 0])
			for model_i, model in enumerate(LOO_models):
				LOO_dr2_dict['delta r2'].extend(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1))
				LOO_dr2_dict['mouse'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				LOO_dr2_dict['left out'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*model)
				dr2_values[well_modelled_idx, model_i] = np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)
			for model_i, model in enumerate(dim_control_models):
				ctrl_LOO_dr2_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1))
				ctrl_LOO_dr2_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				ctrl_LOO_dr2_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*model)
				ctrl_dr2_values[well_modelled_idx, model_i] = np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)
			full_model_r2 = np.mean(r2_full_model, axis=1)
			percent_dr2_LOO = dr2_values/full_model_r2[..., np.newaxis]
			percent_dr2_ctrl = ctrl_dr2_values/full_model_r2[..., np.newaxis]
			for model_i, model in enumerate(LOO_models):
				perc_LOO_dr2_dict['percent delta r2'].extend(percent_dr2_LOO[well_modelled_idx, model_i])
				perc_LOO_dr2_dict['mouse'].extend(np.ones(len(percent_dr2_LOO[well_modelled_idx, model_i]), dtype=object)*mouseID)
				perc_LOO_dr2_dict['left out'].extend(np.ones(len(percent_dr2_LOO[well_modelled_idx, model_i]), dtype=object)*model)
			for model_i, model in enumerate(dim_control_models):
				perc_ctrl_LOO_dr2_dict['percent delta r2'].extend(percent_dr2_ctrl[well_modelled_idx, model_i])
				perc_ctrl_LOO_dr2_dict['mouse'].extend(np.ones(len(percent_dr2_ctrl[well_modelled_idx, model_i]), dtype=object)*mouseID)
				perc_ctrl_LOO_dr2_dict['left out'].extend(np.ones(len(percent_dr2_ctrl[well_modelled_idx, model_i]), dtype=object)*model)

			if include_sign_flip:
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'matched category\nunique')
				
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'mismatched\ncategory unique')
				
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, 4]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 4]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 4]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'shared')

				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_LOO[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, 3]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'sign flip')

			else:
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, 0]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 0]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 0]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'matched category\nunique')
				
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_dim_control[well_modelled_idx, :, 1]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 1]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_dim_control[well_modelled_idx, :, 1]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'mismatched\ncategory unique')
				
				same_size_edges_dict['delta r2'].extend(np.nanmean(r2_LOO[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1))
				same_size_edges_dict['mouse'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*mouseID)
				same_size_edges_dict['left out'].extend(np.ones(len(np.nanmean(r2_LOO[well_modelled_idx, :, 2]-r2_full_model[well_modelled_idx, :], axis=1)), dtype=object)*'shared')

			


			plt.scatter(ctrl_dr2_values[:, 0], dr2_values[:, 2], color='k', alpha=0.3, marker='.')
		plt.plot([0.05, -1.05], [0.05,-1.05], '--', color='grey')
		plt.ylim(-1, 0.1)
		plt.xlim(-1, 0.1)
		plt.xlabel('leave out within-category unique edges, downsampled to n shared edges')
		plt.ylabel('leave out shared edges')
		plt.title(f'{mouseID}')
		#plt.savefig(fig_save_dir + 'within cat unique vs shared LOO d r2 scatterplot.png')
		
		plt.ylim(-.5, 0.01)
		plt.xlim(-.5, 0.01)
		#plt.savefig(fig_save_dir + 'within cat unique vs shared LOO d r2 scatterplot zoomed.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

		by_mouse_LOO_dict[mouseID] = {'delta r2':np.array(LOO_dr2_dict['delta r2'])[np.array(LOO_dr2_dict['mouse'], dtype=object)==mouseID], 'left out':np.array(LOO_dr2_dict['left out'])[np.array(LOO_dr2_dict['mouse'], dtype=object)==mouseID]}
		sns.violinplot(data=by_mouse_LOO_dict[mouseID], x='left out', y='delta r2', cut=0)
		plt.ylim(-1.01, .01)
		plt.title(f'{mouseID} LOO by FC type delta r\u00b2')
		if show_plots:
			plt.show()
		else:
			plt.close()

		dim_matched_by_mouse_LOO_dict[mouseID] = {'delta r2':np.array(same_size_edges_dict['delta r2'])[np.array(same_size_edges_dict['mouse'], dtype=object)==mouseID], 'left out':np.array(same_size_edges_dict['left out'])[np.array(same_size_edges_dict['mouse'], dtype=object)==mouseID]}
		sns.violinplot(data=dim_matched_by_mouse_LOO_dict[mouseID], x='left out', y='delta r2', cut=0)
		plt.ylim(-.2, .025)
		plt.title(f'{mouseID} FC-count matched LOO by FC type delta r\u00b2')
		if show_plots:
			plt.show()
		else:
			plt.close()

	plt.scatter(-np.array(ctrl_LOO_dr2_dict['delta r2'])[np.array(ctrl_LOO_dr2_dict['left out'], dtype=object)=='undersamp_cat_unique_to_shared'], 
		-np.array(LOO_dr2_dict['delta r2'])[np.array(LOO_dr2_dict['left out'], dtype=object)=='shared_edges'],
		color='k', alpha=0.15, marker='.')
	plt.plot([-1, 1], [-1, 1], '--', color='grey')
	plt.ylim(-0.015, 0.3)
	plt.xlim(-0.015, 0.3)
	plt.xlabel('leave out within-category unique FCs, downsampled to n shared FCs')
	plt.ylabel('leave out shared FCs')
	plt.title('all cells, all mice delta r\u00b2')
	#plt.savefig(multimouse_fig_dir + 'downsampled shared vs category unique delta r2 scatterplot.png')
	if show_plots:
		plt.show()
	else:
		plt.close()

	# scatterplot combined across mice but with the color=density thing from Hal
	# but do the gaussian kde only with the points that will actually be in frame
	from scipy.stats import gaussian_kde
	xy = np.vstack([np.array(ctrl_LOO_dr2_dict['delta r2'])[np.array(ctrl_LOO_dr2_dict['left out'], dtype=object)=='undersamp_cat_unique_to_shared'], np.array(LOO_dr2_dict['delta r2'])[np.array(LOO_dr2_dict['left out'], dtype=object)=='shared_edges']])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	LO_cat, LO_shared, z = np.array(ctrl_LOO_dr2_dict['delta r2'])[np.array(ctrl_LOO_dr2_dict['left out'], dtype=object)=='undersamp_cat_unique_to_shared'][idx], np.array(LOO_dr2_dict['delta r2'])[np.array(LOO_dr2_dict['left out'], dtype=object)=='shared_edges'][idx], z[idx]
	best_fit = stats.linregress(-LO_shared, -LO_cat)
	print(best_fit)

	r = stats.pearsonr(-LO_shared, -LO_cat)
	
	fig, ax = plt.subplots()
	ax.scatter(-LO_cat, -LO_shared, c=z, marker='.', cmap='hot', rasterized=True)
	ax.plot([-1, 1], [-1, 1], '--', color='grey')
	ax.plot(best_fit.slope*np.unique(-LO_shared)+best_fit.intercept, np.unique(-LO_shared), color='indianred', lw=2)
	ax.set_ylim(-0.015, 0.3)
	ax.set_xlim(-0.015, 0.3)
	ax.set_xlabel('leave out within-category unique couplings, downsampled to n shared correlations')
	ax.set_ylabel('leave out shared couplings')
	#ax.set_title(f'r={r.statistic:.4f}')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_box_aspect(1)
	#plt.savefig(multimouse_fig_dir + 'downsampled shared vs category unique delta r2 scatterplot colored by density.pdf', dpi=500)
	if show_plots:
		plt.show()
	else:
		plt.close()

	# LOO (full) violin
	# plot as a custom color matplotlib plot (thanks Claude)
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(LOO_dr2_dict)

	# assign colors
	categories = df['left out'].unique()
	colors = ['royalblue', 'indianred', 'rebeccapurple', 'mediumorchid']

	fig, ax = plt.subplots()

	# Build list of data arrays, one per category (in order)
	data_by_cat = [df.loc[df['left out'] == cat, 'delta r2'].values for cat in categories]

	# Plot violins
	parts = ax.violinplot(data_by_cat, positions=range(len(categories)), showmedians=True)

	# Recolor each violin body and its lines
	for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
		pc.set_facecolor(color)
		pc.set_alpha(0.7)

	for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
		if partname in parts:
			parts[partname].set_edgecolor('black')
			parts[partname].set_linewidth(1.5)

	ax.set_xticks(range(len(categories)))
	ax.set_xticklabels(['matched category\nunique', 'mismatched\ncategory unique', 'shared', 'sign flip'])
	ax.set_xlabel('left out')
	ax.set_ylabel('delta r\u00b2')
	#ax.set_title('Delta R2 Compared to Full Model')
	ax.axhline(0, ls='--', color='k')
	ax.set_ylim(-.3, .1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.tight_layout()
	#plt.savefig(multimouse_fig_dir + 'shared unique delta r2s violin.pdf', dpi=500)
	if show_plots:
		plt.show()
	else:
		plt.close()

	
	# plot as a custom color matplotlib plot (thanks Claude)
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(same_size_edges_dict)

	# assign colors
	categories = df['left out'].unique()
	colors = ['royalblue', 'indianred', 'rebeccapurple', 'mediumorchid']

	fig, ax = plt.subplots()

	# Build list of data arrays, one per category (in order)
	data_by_cat = [df.loc[df['left out'] == cat, 'delta r2'].values for cat in categories]

	# Plot violins
	parts = ax.violinplot(data_by_cat, positions=range(len(categories)), showmedians=True)

	# Recolor each violin body and its lines
	for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
		pc.set_facecolor(color)
		pc.set_alpha(0.7)

	for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
		if partname in parts:
			parts[partname].set_edgecolor('black')
			parts[partname].set_linewidth(1.5)

	ax.set_xticks(range(len(categories)))
	ax.set_xticklabels(categories)
	ax.set_xlabel('count-matched left out')
	ax.set_ylabel('delta r\u00b2')
	#ax.set_title('Delta R2 Compared to Full Model')
	ax.axhline(0, ls='--', color='k')
	ax.set_ylim(-.1, .025)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.tight_layout()
	#plt.savefig(multimouse_fig_dir + 'shared unique delta r2s violin dim matched.pdf', dpi=500)
	if show_plots:
		plt.show()
	else:
		plt.close()

if plot_LOO_scatter_by_magnitude:
	LOO_models = ['category_unique_edges', 'other_cat_unique_edges', 'shared_edges', 'sign_flip']
	dim_control_models = ['undersamp_cat_unique_to_shared', 'undersamp_other_cat_unique_to_shared', 'undersamp_cat_unique_to_sign_flip', 
		'undersamp_other_cat_unique_to_sign_flip', 'undersamp_shared_to_sign_flip', 'undersamp_larger_cat_to_other_cat']
	data_dict = {'delta_r2':[], 'average in weight':[], 'FC category':[], 'mouseID':[]}
	weight_mag_dict = {'weight magnitude':[], 'FC category':[]}
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		days = src.IO.get_carry_days(mouseID)
		s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		
		# import unique/shared labels
		edges = src.IO.load_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_weighted_PDF_10000_pearson_corr.pkl')

		single_mouse_data_dict = {'delta_r2':[], 'average in weight':[], 'FC category':[]}
		for cat in ['active', 'empty']:
			if normalized:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges_NORMALIZED.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_NORMALIZED.npy')
				r2_dim_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control_NORMALIZED.npy')
			else:
				r2_full_model = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges.npy')
				r2_LOO = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness.npy')
				r2_dim_control = np.load(save_dir + f'{cat}_kinematics_lag{lag}_and_opposite_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control.npy')
			
			if cat=='active':
				FN = np.load(s2p_fld + 'active_FN_pearson_corr.npy')
				cross_cat_unique_edge_idx = edges['unique_empty']
			else:
				FN = np.load(s2p_fld + 'empty_FN_pearson_corr.npy')
				cross_cat_unique_edge_idx = edges['unique_active']

			n_neurons = FN.shape[0]

			# remove diagonal weights
			FN[np.arange(n_neurons), np.arange(n_neurons)] = 0

			# convert to graph-like boolean
			shared_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
			shared_edge_bool[edges['shared']] = True
			graph_like_shared_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_shared_edge_bool[np.triu_indices(n_neurons, 1)] = shared_edge_bool
			graph_like_shared_edge_bool += graph_like_shared_edge_bool.T

			sign_flip_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
			sign_flip_edge_bool[edges['sign_flip']] = True
			graph_like_sign_flip_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_sign_flip_edge_bool[np.triu_indices(n_neurons, 1)] = sign_flip_edge_bool
			graph_like_sign_flip_edge_bool += graph_like_sign_flip_edge_bool.T

			# index unique edges
			unique_edge_idx = edges[f'unique_{cat}']
			unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
			unique_edge_bool[unique_edge_idx] = True
			n_unique_edges = len(unique_edge_idx)
			# and convert to graph-like boolean
			graph_like_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_unique_edge_bool[np.triu_indices(n_neurons, 1)] = unique_edge_bool
			graph_like_unique_edge_bool += graph_like_unique_edge_bool.T

			# index other-cat unique edges
			cross_cat_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
			cross_cat_unique_edge_bool[cross_cat_unique_edge_idx] = True
			# and convert to graph-like boolean
			graph_like_cross_cat_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_cross_cat_unique_edge_bool[np.triu_indices(n_neurons, 1)] = cross_cat_unique_edge_bool
			graph_like_cross_cat_unique_edge_bool += graph_like_cross_cat_unique_edge_bool.T

			weight_mag_dict['weight magnitude'].extend(FN[graph_like_unique_edge_bool].flatten())
			weight_mag_dict['FC category'].extend(np.ones(len(FN[graph_like_unique_edge_bool].flatten()), dtype=object)*'matched unique')
			weight_mag_dict['weight magnitude'].extend(FN[graph_like_cross_cat_unique_edge_bool].flatten())
			weight_mag_dict['FC category'].extend(np.ones(len(FN[graph_like_cross_cat_unique_edge_bool].flatten()), dtype=object)*'mismatched unique')
			weight_mag_dict['weight magnitude'].extend(FN[graph_like_shared_edge_bool].flatten())
			weight_mag_dict['FC category'].extend(np.ones(len(FN[graph_like_shared_edge_bool].flatten()), dtype=object)*'shared')
			weight_mag_dict['weight magnitude'].extend(FN[graph_like_sign_flip_edge_bool].flatten())
			weight_mag_dict['FC category'].extend(np.ones(len(FN[graph_like_sign_flip_edge_bool].flatten()), dtype=object)*'sign flip')

			dr2_values = np.zeros((r2_full_model.shape[0], len(LOO_models)))*np.nan
			ctrl_dr2_values = np.zeros((r2_full_model.shape[0], len(dim_control_models)))*np.nan
			well_modelled_idx = ~np.isnan(r2_LOO[:, 0, 0])
			
			for model_i, model in enumerate(LOO_models):
				dr2_values[well_modelled_idx, model_i] = np.nanmean(r2_LOO[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)

			for model_i, model in enumerate(dim_control_models):
				ctrl_dr2_values[well_modelled_idx, model_i] = np.nanmean(r2_dim_control[well_modelled_idx, :, model_i]-r2_full_model[well_modelled_idx, :], axis=1)

			data_dict['mouseID'].extend(np.ones(np.sum(well_modelled_idx)*4, dtype=object)*mouseID)
			for neuron in np.where(well_modelled_idx)[0]:
				neuron_in_weights = FN[neuron, :]

				# matched unique LOO (downsampled to sign flip)
				# get mean in-weight
				matched_unique_mean_in_weight = np.nanmean(np.abs(neuron_in_weights[graph_like_unique_edge_bool[neuron, :]]))
				# grab LOO dr2 and add to dictionary
				data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 2])
				data_dict['average in weight'].append(matched_unique_mean_in_weight)
				data_dict['FC category'].append('matched unique')
				single_mouse_data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 2])
				single_mouse_data_dict['average in weight'].append(matched_unique_mean_in_weight)
				single_mouse_data_dict['FC category'].append('matched unique')

				# mismatched unique LOO (downsampled to sign flip)
				# get mean in-weight
				mismatched_unique_mean_in_weight = np.nanmean(np.abs(neuron_in_weights[graph_like_cross_cat_unique_edge_bool[neuron, :]]))
				# grab LOO dr2 and add to dictionary
				data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 3])
				data_dict['average in weight'].append(mismatched_unique_mean_in_weight)
				data_dict['FC category'].append('mismatched unique')
				single_mouse_data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 3])
				single_mouse_data_dict['average in weight'].append(mismatched_unique_mean_in_weight)
				single_mouse_data_dict['FC category'].append('mismatched unique')

				# shared LOO (downsampled to sign flip)
				# get mean in-weight
				shared_mean_in_weight = np.nanmean(np.abs(neuron_in_weights[graph_like_shared_edge_bool[neuron, :]]))
				# grab LOO dr2 and add to dictionary
				data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 4])
				data_dict['average in weight'].append(shared_mean_in_weight)
				data_dict['FC category'].append('shared')
				single_mouse_data_dict['delta_r2'].append(ctrl_dr2_values[neuron, 4])
				single_mouse_data_dict['average in weight'].append(shared_mean_in_weight)
				single_mouse_data_dict['FC category'].append('shared')

				# sign flip LOO
				# get mean in-weight
				sign_flip_mean_in_weight = np.nanmean(np.abs(neuron_in_weights[graph_like_sign_flip_edge_bool[neuron, :]]))
				# grab LOO dr2 and add to dictionary
				data_dict['delta_r2'].append(dr2_values[neuron, 3])
				data_dict['average in weight'].append(sign_flip_mean_in_weight)
				data_dict['FC category'].append('sign flip')
				single_mouse_data_dict['delta_r2'].append(dr2_values[neuron, 3])
				single_mouse_data_dict['average in weight'].append(sign_flip_mean_in_weight)
				single_mouse_data_dict['FC category'].append('sign flip')
		
		# plot scatterplot per mouse
		# Convert dict to DataFrame if needed
		df = pd.DataFrame(single_mouse_data_dict)

		# assign colors
		categories = df['FC category'].unique()
		colors = ['royalblue', 'indianred', 'rebeccapurple', 'mediumorchid']

		# Build list of data arrays, one per category (in order)
		d2_by_cat = [df.loc[df['FC category'] == cat, 'delta_r2'].values for cat in categories]
		in_weight_by_cat = [df.loc[df['FC category'] == cat, 'average in weight'].values for cat in categories]

		fig, ax = plt.subplots()
		for cat_i, category in enumerate(categories):
			ax.scatter(in_weight_by_cat[cat_i], d2_by_cat[cat_i], marker='.', color=colors[cat_i], label=category, alpha=0.5)
		print(mouseID)
		for cat_i, category in enumerate(categories):
			best_fit = stats.linregress(in_weight_by_cat[cat_i], d2_by_cat[cat_i])
			print(category)
			print(best_fit)
			r = stats.pearsonr(in_weight_by_cat[cat_i], d2_by_cat[cat_i])
			print(r)
			print('')
			ax.plot(np.unique(in_weight_by_cat[cat_i]), best_fit.slope*np.unique(in_weight_by_cat[cat_i]) + best_fit.intercept, color=colors[cat_i], lw=2)

		print('')
		ax.axhline(0, ls='--', color='k')
		ax.legend()
		ax.set_xlabel('mean FC magnitude removed')
		ax.set_ylabel('\u0394 r\u00b2')
		ax.set_title(f'{mouseID}')
		#plt.savefig(fig_save_dir + 'shared unique delta r2s scatterplot magnitude and FC type.png')
		plt.close()

	# plot combined across mice
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(data_dict)

	# assign colors
	categories = df['FC category'].unique()
	colors = ['royalblue', 'indianred', 'rebeccapurple', 'mediumorchid']

	# Build list of data arrays, one per category (in order)
	d2_by_cat = [df.loc[df['FC category'] == cat, 'delta_r2'].values for cat in categories]
	in_weight_by_cat = [df.loc[df['FC category'] == cat, 'average in weight'].values for cat in categories]

	fig, ax = plt.subplots()
	for cat_i, category in enumerate(categories):
		ax.scatter(in_weight_by_cat[cat_i], d2_by_cat[cat_i], marker='.', color=colors[cat_i], label=category, alpha=0.25, rasterized=True)
	colors = ['mediumblue', 'brown', 'indigo', 'darkorchid']
	# for cat_i, category in enumerate(categories):
	# 	best_fit = stats.linregress(in_weight_by_cat[cat_i], d2_by_cat[cat_i])
	# 	print(category)
	# 	print(best_fit)
	# 	r = stats.pearsonr(in_weight_by_cat[cat_i], d2_by_cat[cat_i])
	# 	print(r)
	# 	ax.plot(np.unique(in_weight_by_cat[cat_i]), best_fit.slope*np.unique(in_weight_by_cat[cat_i]) + best_fit.intercept, color=colors[cat_i], lw=2)
	# 	print('')
	# in_weight_by_cat.pop(1)
	# d2_by_cat.pop(1)
	# best_fit = stats.linregress(np.concatenate(in_weight_by_cat), np.concatenate(d2_by_cat))
	# r = stats.pearsonr(np.concatenate(in_weight_by_cat), np.concatenate(d2_by_cat))
	# print(best_fit)
	# print(r)
	# ax.plot(np.unique(np.concatenate(in_weight_by_cat)), best_fit.slope*np.unique(np.concatenate(in_weight_by_cat)) + best_fit.intercept, color='k', lw=2)
	ax.axhline(0, ls='--', color='k')
	ax.legend()
	ax.set_xlabel('mean FC magnitude removed')
	ax.set_ylabel('\u0394 r\u00b2')
	ax.set_title('FC count-matched LOO models')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig(multimouse_fig_dir + 'shared unique delta r2s scatterplot magnitude and FC type.pdf', dpi=500)
	if show_plots:	
		plt.show()
	else:
		plt.close()

	colors = ['royalblue', 'indianred', 'rebeccapurple', 'mediumorchid']

	fig, ax = plt.subplots()
	for cat_i, category in enumerate(categories):
		ax.hist(in_weight_by_cat[cat_i], color=colors[cat_i], label=category, alpha=0.3, density=True)
		ax.axvline(np.mean(in_weight_by_cat[cat_i]), color=colors[cat_i])
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel('mean correlation magnitude per neuron')
	ax.set_ylabel('probability density')
	#plt.savefig(multimouse_fig_dir + 'FC in weight density histogram.png', dpi=500)
	if show_plots:	
		plt.show()
	else:
		plt.close()

	fig, ax = plt.subplots()
	for cat_i, category in enumerate(categories):
		ax.hist(in_weight_by_cat[cat_i], color=colors[cat_i], label=category, alpha=0.3)
		ax.axvline(np.mean(in_weight_by_cat[cat_i]), color=colors[cat_i])
	ax.legend()
	ax.set_xlabel('mean correlation magnitude per neuron')
	ax.set_ylabel('neuron count')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	#plt.savefig(multimouse_fig_dir + 'FC in weight count histogram.png', dpi=500)
	if show_plots:	
		plt.show()
	else:
		plt.close()

	df = pd.DataFrame(weight_mag_dict)
	weight_magnitudes = [df.loc[df['FC category'] == cat, 'weight magnitude'].values for cat in categories]
	
	fig, ax = plt.subplots()
	for cat_i, category in enumerate(categories):
		ax.hist(np.abs(weight_magnitudes[cat_i]), color=colors[cat_i], label=category, alpha=0.3, density=True)
		ax.axvline(np.mean(np.abs(weight_magnitudes[cat_i])), color=colors[cat_i])
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel('correlation magnitude')
	ax.set_ylabel('probability density')
	#plt.savefig(multimouse_fig_dir + 'FC by weight density histogram.png', dpi=500)
	if show_plots:	
		plt.show()
	else:
		plt.close()

	fig, ax = plt.subplots()
	for cat_i, category in enumerate(categories):
		ax.hist(np.abs(weight_magnitudes[cat_i]), color=colors[cat_i], label=category, alpha=0.3)
		ax.axvline(np.mean(np.abs(weight_magnitudes[cat_i])), color=colors[cat_i])
	ax.legend()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xlabel('correlation magnitude')
	ax.set_ylabel('functional connection count')
	#plt.savefig(multimouse_fig_dir + 'FC by weight count histogram.png', dpi=500)
	if show_plots:	
		plt.show()
	else:
		plt.close()


if permutations:
	for mouseID in mice:
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir = save_dir+'figures/'
		r2_dict = {'mouse':[], 'r2':[], 'permutation':[]}
		for category in ['active', 'empty']:
			if normalized:
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
				weights_edges_permute = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy')
				weights_permute = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy')
				no_weights = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_no_weights_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy')
				opp_cat_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				weights_edges_permute = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy')
				weights_permute = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy')
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
				no_weights = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_no_weights_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy')
				opp_cat_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{n_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')


			weights_edges_median = np.median(np.nanmean(weights_edges_permute, axis=2), axis=1)
			weights_median = np.median(np.nanmean(weights_permute, axis=2), axis=1)
			full_median = np.median(full_model_r2, axis=1)
			no_weights_median = np.median(no_weights, axis=1)
			opp_cat_median = np.median(opp_cat_r2, axis=1)

			r2_dict['r2'].extend(full_median)
			r2_dict['r2'].extend(weights_median)
			r2_dict['r2'].extend(weights_edges_median)
			r2_dict['r2'].extend(no_weights_median)
			r2_dict['mouse'].extend(np.ones(len(full_median)*4, dtype=object)*mouseID)
			r2_dict['permutation'].extend(np.ones(len(full_median), dtype=object)*'full FN')
			r2_dict['permutation'].extend(np.ones(len(full_median), dtype=object)*'permuted weights')
			r2_dict['permutation'].extend(np.ones(len(full_median), dtype=object)*'permuted weights\nand connectivity')
			r2_dict['permutation'].extend(np.ones(len(full_median), dtype=object)*'no weights\n(summed\npopulation activity)')

	# plot as a custom color matplotlib plot (thanks Claude)
	# Convert dict to DataFrame if needed
	df = pd.DataFrame(r2_dict)

	# assign colors
	categories = df['permutation'].unique()

	fig, ax = plt.subplots()

	# Build list of data arrays, one per category (in order)
	data_by_cat = [df.loc[df['permutation'] == cat, 'r2'].values for cat in categories]

	# Plot violins
	parts = ax.violinplot(data_by_cat, positions=[0, 1, 2, 3], showmedians=True)

	for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
		if partname in parts:
			parts[partname].set_edgecolor('black')
			parts[partname].set_linewidth(1.5)

	full_vs_weights = stats.ttest_rel(data_by_cat[0], data_by_cat[1], alternative='greater')
	full_vs_weights_and_edges = stats.ttest_rel(data_by_cat[0], data_by_cat[2], alternative='greater')
	full_vs_no_FN = stats.ttest_rel(data_by_cat[0], data_by_cat[3], alternative='greater')
	weights_vs_weights_and_edges = stats.ttest_rel(data_by_cat[1], data_by_cat[2], alternative='greater')
	print(full_vs_weights)
	print(full_vs_weights_and_edges)
	print(full_vs_no_FN)
	print(weights_vs_weights_and_edges)


	ax.plot([0, 1], np.ones(2)*(np.max(data_by_cat[0])+.025), color='k')
	ax.plot([0, 2], np.ones(2)*(np.max(data_by_cat[0])+.075), color='k')
	ax.plot([0, 3], np.ones(2)*(np.max(data_by_cat[0])+.125), color='k')
	ax.plot([1, 2], np.ones(2)*(np.max(data_by_cat[1])+.025), color='k')
	if full_vs_weights.pvalue<.001:
		ax.text(.5, np.max(data_by_cat[0])+.03, '***', horizontalalignment='center')
	elif full_vs_weights.pvalue<.01:
		ax.text(.5, np.max(data_by_cat[0])+.03, '**', horizontalalignment='center')
	elif full_vs_weights.pvalue<.05:
		ax.text(.5, np.max(data_by_cat[0])+.03, '*', horizontalalignment='center')
	else:
		ax.text(.5, np.max(data_by_cat[0])+.03, 'n.s.', horizontalalignment='center')

	if full_vs_weights_and_edges.pvalue<.001:
		ax.text(1, np.max(data_by_cat[0])+.08, '***', horizontalalignment='center')
	elif full_vs_weights_and_edges.pvalue<.01:
		ax.text(1, np.max(data_by_cat[0])+.08, '**', horizontalalignment='center')
	elif full_vs_weights_and_edges.pvalue<.05:
		ax.text(1, np.max(data_by_cat[0])+.08, '*', horizontalalignment='center')
	else:
		ax.text(1, np.max(data_by_cat[0])+.08, 'n.s.', horizontalalignment='center')

	if full_vs_no_FN.pvalue<.001:
		ax.text(1.5, np.max(data_by_cat[0])+.13, '***', horizontalalignment='center')
	elif full_vs_no_FN.pvalue<.01:
		ax.text(1.5, np.max(data_by_cat[0])+.13, '**', horizontalalignment='center')
	elif full_vs_no_FN.pvalue<.05:
		ax.text(1.5, np.max(data_by_cat[0])+.13, '*', horizontalalignment='center')
	else:
		ax.text(1.5, np.max(data_by_cat[0])+.13, 'n.s.', horizontalalignment='center')

	if weights_vs_weights_and_edges.pvalue<.001:
		ax.text(1.5, np.max(data_by_cat[1])+.03, '***', horizontalalignment='center')
	elif weights_vs_weights_and_edges.pvalue<.01:
		ax.text(1.5, np.max(data_by_cat[1])+.03, '**', horizontalalignment='center')
	elif weights_vs_weights_and_edges.pvalue<.05:
		ax.text(1.5, np.max(data_by_cat[1])+.03, '*', horizontalalignment='center')
	else:
		ax.text(1.5, np.max(data_by_cat[1])+.03, 'n.s.', horizontalalignment='center')

	ax.set_xticks([0, 1, 2, 3])
	ax.set_xticklabels(categories)
	#ax.set_xlabel('permutation')
	ax.set_ylabel('r\u00b2')
	#ax.set_title('Delta R2 Compared to Full Model')
	ax.axhline(0, ls='--', color='dimgrey')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	plt.tight_layout()
	plt.savefig(multimouse_fig_dir + 'encoding permutation comparison.pdf', dpi=500)
	if show_plots:
		plt.show()
	else:
		plt.close()