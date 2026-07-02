'''
encoding model that combines active and empty into a single model
y = kinematics + shared couplings + unique active grasp couplings + unique empty grasp couplings
use the same train/test splits as the base encoding model
'''
import numpy as np
import src.utils
import src.IO
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
import multiprocessing
import functools
import warnings
from pathlib import Path
data_dir = Path.cwd() / 'data'

def ridge_regression_worker(neuron, which_fold, active_features, empty_features, active_neural_data, empty_neural_data, shared_graph_bool, active_unique_graph_bool, empty_unique_graph_bool, active_cv_splits, empty_cv_splits, fit_params=False):
	'''
	worker function for parallel processing
	does one run of the encoding model for a single neuron within a specific fold
	returns the test r^2 value on that fold for that neuron
	'''

	n_neurons = active_neural_data.shape[0]

	active_training_idx, active_testing_idx = active_cv_splits[neuron][which_fold][0], active_cv_splits[neuron][which_fold][1]
	empty_training_idx, empty_testing_idx = empty_cv_splits[neuron][which_fold][0], empty_cv_splits[neuron][which_fold][1]

	combined_features_training_pre_coupling = np.vstack((active_features[active_training_idx, :], empty_features[empty_training_idx, :]))
	combined_features_testing_pre_coupling = np.vstack((active_features[active_testing_idx, :], empty_features[empty_testing_idx, :]))
	combined_neural_training_data = np.hstack((active_neural_data[:, active_training_idx], empty_neural_data[:, empty_training_idx]))
	combined_neural_testing_data = np.hstack((active_neural_data[:, active_testing_idx], empty_neural_data[:, empty_testing_idx]))

	training_time = combined_features_training_pre_coupling.shape[0]
	testing_time = combined_features_testing_pre_coupling.shape[0]

	# calculate the coupling terms
	# first find the weights based on the training data
	# for active unique edges
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(active_neural_data[:, active_training_idx])[neuron, :]
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	weights_onto_neuron[~active_unique_graph_bool[neuron, :]] = 0
	weights_onto_neuron[neuron] = 0

	# then add per timepoint to get weighted input
	active_unique_coupling_term_training = weights_onto_neuron @ combined_neural_training_data
	active_unique_coupling_term_testing = weights_onto_neuron @ combined_neural_testing_data


	# empty unique edges
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(empty_neural_data[:, empty_training_idx])[neuron, :]
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	weights_onto_neuron[~empty_unique_graph_bool[neuron, :]] = 0
	weights_onto_neuron[neuron] = 0

	# then add per timepoint to get weighted input
	empty_unique_coupling_term_training = weights_onto_neuron @ combined_neural_training_data
	empty_unique_coupling_term_testing = weights_onto_neuron @ combined_neural_testing_data


	# then combined for shared edges
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(combined_neural_training_data)[neuron, :]
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	weights_onto_neuron[~shared_graph_bool[neuron, :]] = 0
	weights_onto_neuron[neuron] = 0

	# then add per timepoint to get weighted input
	shared_coupling_term_training = weights_onto_neuron @ combined_neural_training_data
	shared_coupling_term_testing = weights_onto_neuron @ combined_neural_testing_data


	# make the combined feature space
	training_feats = np.hstack((combined_features_training_pre_coupling, active_unique_coupling_term_training[:, np.newaxis], empty_unique_coupling_term_training[:, np.newaxis], shared_coupling_term_training[:, np.newaxis]))
	testing_feats = np.hstack((combined_features_testing_pre_coupling, active_unique_coupling_term_testing[:, np.newaxis], empty_unique_coupling_term_testing[:, np.newaxis], shared_coupling_term_testing[:, np.newaxis]))

	# normalize features to the same relative scales (at least in training) - maximum normalization
	max_feats = np.max(training_feats, axis=0)
	if any(max_feats==0):
		zero_idx = np.where(max_feats==0)
		max_feats[zero_idx[0]]=1
	training_feats = training_feats/max_feats
	testing_feats = testing_feats/max_feats

	if fit_params:
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(training_feats, combined_neural_training_data[neuron, :])
		alpha=search.best_params_['alpha']
	else:
		alpha = features.shape[0]/8

	regression = Ridge(alpha = alpha)
	regression.fit(training_feats, combined_neural_training_data[neuron, :])
	r2 = regression.score(testing_feats, combined_neural_testing_data[neuron, :])
	betas = regression.coef_
	return r2, betas

def parallelized_ridge_regression(active_features, empty_features, active_neural_data, empty_neural_data, shared_graph_bool, active_unique_graph_bool, empty_unique_graph_bool, active_cv_splits, empty_cv_splits, n_cv_folds):
	'''
	calculates and returns the testing r^2 values and model betas for all folds, all neurons 
	performs regressions parallelized across neurons
	'''
	n_neurons = active_neural_data.shape[0]
	print(n_neurons)
	n_features = active_features.shape[1]+3
	r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	betas = np.zeros((n_neurons, n_features, n_cv_folds))*np.nan
	
	for fold in range(n_cv_folds):
		print(f'working on fold {fold}...')
		worker_func = functools.partial(
		ridge_regression_worker,
		which_fold=fold, 
		active_features=active_features, 
		empty_features=empty_features, 
		active_neural_data=active_neural_data, 
		empty_neural_data=empty_neural_data, 
		shared_graph_bool=shared_graph_bool, 
		active_unique_graph_bool=active_unique_graph_bool, 
		empty_unique_graph_bool=empty_unique_graph_bool, 
		active_cv_splits=active_cv_splits, 
		empty_cv_splits=empty_cv_splits, 
		fit_params=True 
		)
	
		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a list of r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		# unpack the tuple of results into the betas and r2 scores
		fold_r2_scores = []
		fold_betas = []
		for tup in results:
			r2, beta = tup
			fold_r2_scores.append(r2)
			fold_betas.append(beta)

		r2_scores[:, fold] = np.array(fold_r2_scores)
		betas[:, :, fold] = np.array(fold_betas)

	return r2_scores, betas


mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

t_pre = 4 # neural frames relative to rotation time
t_post = 4 # neural frames relative to rotation time
time_in_trial = t_pre + t_post # neural frames

splitter = 'KFold' # KFold or Shuffle, if KFold, then cv_folds sets the train/test split, if Shuffle then 70/30
split_method = 'trial'


if splitter=='Shuffle':
	cv_folds=10
	test_size=0.3
elif splitter=='KFold':
	cv_folds=2
	n_repeats=10
lag=int(2)

if split_method=='trial':
	for mouseID in mice:
		# load in data
		days = src.IO.get_carry_days(mouseID)
		drive = data_dir + '/neural'
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		
		reg_inds = src.utils.load_registered_cells(mouse_dir, days)

		neural_activity_by_trial = [[] for day in days]
		behavior = [[] for day in days]
		day_labels = [[] for day in days]
		trial_labels = [[] for day in days]
		trial_like_day_labels = [[] for day in days]
		trial_like_cat_labels = [[] for day in days]
		trial_counter=0
		for i, day in enumerate(days):
			print(day)
			# load in the Cascade spikes
			calcium_data_path = mouse_dir + day
			s2p_fld = calcium_data_path + '/'

			spks = np.load(s2p_fld + 'cascade_spks.npy')
			
			# index by the registered cells
			reg_spks = spks[reg_inds[i], :]
			n_neurons = reg_spks.shape[0]

			# load in the behavioral times
			carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
			carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
			carry_times = carry_times[carry_labels!=0]
			carry_labels = carry_labels[carry_labels!=0]-1
			trial_like_cat_labels[i] = carry_labels
			category_labels_by_time = np.zeros((len(carry_times)*time_in_trial))

			# load in the kinematic data
			paw_centroid = np.load(s2p_fld+'paw_centroid_interpolated.npy')
			aperature = np.load(s2p_fld+'aperature_interpolated_conservative.npy')
			splay = np.load(s2p_fld+'d2d3_splay_interpolated_conservative.npy')
			orientation = np.load(s2p_fld+'relative_y_normal_to_palm_interpolated.npy')
			# and combine
			kinematic_data = np.vstack((paw_centroid, aperature, splay, orientation)) # now shape=(K, time) where K=6 kinematic variables

			# get trial-specific neural data and behavioral data
			temp_spks = [[] for time in carry_times]
			temp_behavior = [[] for time in carry_times]
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry+t_post)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_labels[i].extend(np.ones(time_in_trial)*trial_counter)
				trial_counter+=1
			neural_activity_by_trial[i] = np.hstack(temp_spks)
			behavior[i] = np.hstack(temp_behavior)
			behavior[i] = np.vstack((behavior[i], category_labels_by_time))
			day_labels[i] = np.ones(behavior[i].shape[1])*i
			trial_like_day_labels[i] = np.ones(len(carry_times))*i

		# combine across days
		neural_activity = np.hstack(neural_activity_by_trial)
		behavior = np.hstack(behavior).T

		day_labels = np.hstack(day_labels)
		trial_labels = np.hstack(trial_labels)
		trial_like_day_labels = np.hstack(trial_like_day_labels)
		trial_like_cat_labels = np.hstack(trial_like_cat_labels)
		print(day_labels.shape, trial_labels.shape, behavior.shape)

		# remove neurons with only NaN for activity
		activity_nonan = neural_activity[~np.isnan(neural_activity).any(axis=1), :]
		print(np.sum(np.isnan(neural_activity).any(axis=1)), ' NaN-containing cells removed')
		NaN_indices = np.where(np.isnan(neural_activity).any(axis=1))[0]
		red_cells_bool = red_cells[~np.isnan(neural_activity).any(axis=1)]
		# print(NaN_indices)
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			# print(np.where(where_not_NaN)[0])
			# print(np.unique(trial_labels))
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		# grab unique and shared indices
		shared_unique = src.IO.load_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_weighted_PDF_10000_pearson_corr.pkl')
		print('n unique active edges: ', len(shared_unique['unique_active']))
		print('n unique empty edges: ', len(shared_unique['unique_empty']))
		print('n shared edges: ', len(shared_unique['shared']))
		print('n sign flip edges: ', len(shared_unique['sign_flip']))

		# convert to graph-like boolean
		shared_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
		shared_edge_bool[shared_unique['shared']] = True
		graph_like_shared_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
		graph_like_shared_edge_bool[np.triu_indices(n_neurons, 1)] = shared_edge_bool
		graph_like_shared_edge_bool += graph_like_shared_edge_bool.T

		active_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
		active_unique_edge_bool[shared_unique['unique_active']] = True
		graph_like_active_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
		graph_like_active_unique_edge_bool[np.triu_indices(n_neurons, 1)] = active_unique_edge_bool
		graph_like_active_unique_edge_bool += graph_like_active_unique_edge_bool.T

		empty_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
		empty_unique_edge_bool[shared_unique['unique_empty']] = True
		graph_like_empty_unique_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
		graph_like_empty_unique_edge_bool[np.triu_indices(n_neurons, 1)] = empty_unique_edge_bool
		graph_like_empty_unique_edge_bool += graph_like_empty_unique_edge_bool.T

		# separate out active and empty trial data for indexing by train/test fold labels
		active_features = features[behavior[where_not_NaN, -1]==0, :]
		active_neural_data = neuronal_data[:, behavior[where_not_NaN, -1]==0]
		empty_features = features[behavior[where_not_NaN, -1]==1, :]
		empty_neural_data = neuronal_data[:, behavior[where_not_NaN, -1]==1]

		if splitter=='Shuffle':
			# import cv split indices
			active_cv_split_indices = src.IO.load_pickle(save_dir + f'active_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
			empty_cv_split_indices = src.IO.load_pickle(save_dir + f'empty_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
			total_n_folds = cv_folds
		elif splitter=='KFold':
			# import cv split indices
			active_cv_split_indices = src.IO.load_pickle(save_dir + f'active_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
			empty_cv_split_indices = src.IO.load_pickle(save_dir + f'empty_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
			total_n_folds = cv_folds*n_repeats

		r2_values, betas = parallelized_ridge_regression(active_features, empty_features, active_neural_data, empty_neural_data, graph_like_shared_edge_bool, graph_like_active_unique_edge_bool, graph_like_empty_unique_edge_bool, active_cv_split_indices, empty_cv_split_indices, total_n_folds)
		cross_fold_mean_betas = np.mean(betas, axis=2)	
		
		print(mouseID)
		print('median full model r2: ', np.median(np.mean(r2_values, axis=1)))
		print('active unique median beta: ', np.median(cross_fold_mean_betas[:, -3]))
		print('empty unique median beta: ', np.median(cross_fold_mean_betas[:, -2]))
		print('shared median beta: ', np.median(cross_fold_mean_betas[:, -1]))
		print()

		# save out per mouse
		np.save(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values_normalized_features.npy', r2_values)
		np.save(save_dir + f'combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_betas_normalized_features.npy', betas)
