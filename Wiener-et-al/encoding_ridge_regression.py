'''
ridge regression encoding models
the base model is:
	r_i^k(t) = beta_0+beta_1*c_x(t)+beta_2*c_y(t)+beta_3*c_z(t)+beta_4*s(t)+beta_5*a(t)+beta_6*o(t)+beta_7*W_i^k.r^k(t)
	where:
		r_i^k(t) is the activity of neuron i during condition k
		c_x(t), c_y(t), and c_z(t) are the x, y, and z position of the paw centroid
		s(t) is splay
		a(t) is aperture
		o(t) is orientation
		W_i^k.r^k(t) is a coupling term for the weighted "input": the dot product of a functional network and population activity of other neurons
			except for condition mismatched, W_i^k is calculated from the training data

variations on this model include:
	- kinematics only (no coupling term)
	- condition matched (FN k  is used to predict r_i^k) --> e.g. active FN, predicting active activity
	- condition mismatched (FN c is used to predict r_i^k) --> e.g. empty FN, predicting active activity
	- FC type-based LOOs:
		- all matched unique, mismatched unique, shared, or sign flip FCs are set to 0 in the FN
		- a FC-count matched subset of matched unique, mismatched unique, shared, or sign flip FCs are set to 0 in the FN
	- FC weight strength-based LOOs:
		- all strong, medium, weak, and zero magnitude FCs (based on percentile) are set to 0 in the FN
		- a FC-count matched subset of trong, medium, and weak FCs (based on percentile) are set to 0 in the FN
	- no weights (W_i^k is 1 such that the coupling term is just the sum of all neurons except the predicted neuron)
	- permutations:
		- repeated random reassignment of FC weights (magnitude and sign) among all present FCs (preserves topology of the network)
		- repeated random reassignment of FC weights (magnitude and sign) among all possible FCs (disrupts topology of the network)

this code additionally generates the train/test splits that all of these models will use. 
To replicate the figures from the paper exactly, use calculate_splits=False and use the cv split pickles in the data folder.
'''
import numpy as np
import src.utils
import matplotlib.pyplot as plt
import os
import multiprocessing
import functools
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import Ridge
import warnings
from scipy import stats
from collections import Counter
import random

warnings.filterwarnings("ignore", module="sklearn")

# Per-process state populated by Pool initializer. Lets us keep large invariants
# (features, neural_data, cv_splits, weights_list, masks list) in worker memory
# instead of re-pickling them on every pool.map call.
_W = {}

def _pool_init(state):
	_W.update(state)

def _within_cat_flat(task):
	'''state initializer for category matched regression'''
	neuron, fold, mask_idx = task
	return within_cat_ridge_regression_worker(
		neuron, fold,
		_W['features'], _W['neural_data'],
		_W['masks'][mask_idx], _W['cv_splits'],
		_W.get('fit_params', False), _W.get('normalized', True),
	)

def _dim_control_flat(task):
	'''state initializer for FC-count matched LOO models'''
	neuron, fold, mask_idx = task
	return dim_control_ridge_regression_worker(
		neuron, fold,
		_W['features'], _W['neural_data'],
		_W['weights_list'], _W['masks'][mask_idx],
		_W['cv_splits'],
		_W.get('fit_params', False), _W.get('normalized', True),
	)

def _weight_calc_flat(task):
	'''state initializer for calculating FNs from training data'''
	neuron, fold = task
	return weight_calculation_worker(neuron, _W['cv_splits'], fold, _W['neural_data'])

def _permutation_flat(task):
	'''state initializer for FN permutation models'''
	perm_idx, fold, neuron, nulls_permute = task
	return permutation_ridge_regression_worker(
		perm_idx, fold, neuron,
		_W['features'], _W['neural_data'],
		_W['weights_per_neuron'][neuron], _W['graph_bool'],
		_W['cv_splits'],
		_W.get('fit_params', False), _W.get('normalized', True),
		nulls_permute,
	)

def within_cat_ridge_regression_worker(neuron, which_fold, features, neural_data, graph_bool, cv_splits, fit_params=False, normalized=True):
	'''
	worker function for parallel processing
	does one run of the category-matched encoding model for a single neuron within a specific fold
	returns the test r^2 value on that fold for that neuron
	'''
	# grab training and testing data for a specific neuron and specific CV fold
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]

	# calculate the FN from the training data
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(neural_data[:, training_idx])[neuron, :]
	# convert any NaNs (from constant data in the training data) to 0
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	# manually set all null FCs to 0
	weights_onto_neuron[~graph_bool[neuron, :]] = 0
	# set the neuron's correlation with itself (always 1) to 0 so that neuron data is not used to predict itself
	weights_onto_neuron[neuron] = 0

	# take the dot product of the FN and neuronal data as the coupling term
	coupling_term = weights_onto_neuron @ neural_data
	# assemble the feature space
	feats = np.hstack((features, coupling_term[:, np.newaxis]))
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]

	if normalized:
		# normalize all feature data by the maximum of the training data	
		feat_maxes = np.max(feats[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = feats/feat_maxes

	if fit_params:
		# fit alpha parameter (ridge regularization parameter) on training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# or just take the number of trials/2
		alpha = features.shape[0]/16

	# fit the regression
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate testing r^2
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	return r2

def opposite_cat_ridge_regression_worker(neuron, which_fold, features, neural_data, graph_like_weights, cv_splits, fit_params=False, normalized=False):
	'''
	worker function for parallel processing
	does one run of the category-mismatched encoding model for a single neuron within a specific fold
	returns the test r^2 value on that fold for that neuron
	'''
	# grab training and testing splits for this CV fold for this neuron
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]
	# take the FN as the pre-calculated FN from the opposite trial category 
	# (this has already been processed to remove the diagonal and all null FCs)
	weights_onto_neuron = graph_like_weights[neuron, :]
	# calculate the coupling term by taking the dot product of the FN and neuronal activity
	coupling_term = weights_onto_neuron @ neural_data
	# assemble the feature space
	feats = np.hstack((features, coupling_term[:, np.newaxis]))
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]
	
	if normalized:
		# normalize all feature data by the maximum of the training data	
		feat_maxes = np.max(feats[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = feats/feat_maxes

	if fit_params:
		# fit alpha parameter (ridge regularization parameter) on training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# or just take the number of trials/2
		alpha = features.shape[0]/16

	# fit the regression
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate testing r^2
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	return r2

def drop_applied_within_cat_ridge_regression_worker(neuron, which_fold, features, neural_data, graph_bool, predrop_behavior, predrop_neural_data, postdrop_behavior, postdrop_neural_data, cv_splits, fig_save_dir=None, fit_params=False, normalized=False):
	'''
	worker function for parallel processing
	does one run of training the regression then applying it on drop timepoints
	returns the test r^2 value on that fold for that neuron within-category, for pre-drop timepoints, and for post-drop timepoints
	'''
	# grab training and testing splits for this CV fold for this neuron
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]

	# calculate the FN from the training data
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(neural_data[:, training_idx])[neuron, :]
	# convert any NaNs (from constant data in the training data) to 0
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	# manually set all null FCs to 0
	weights_onto_neuron[~graph_bool[neuron, :]] = 0
	# set the neuron's correlation with itself (always 1) to 0 so that neuron data is not used to predict itself
	weights_onto_neuron[neuron] = 0

	# take the dot product of the FN and neuronal data as the coupling term
	coupling_term = weights_onto_neuron @ neural_data
	# assemble the feature space
	feats = np.hstack((features, coupling_term[:, np.newaxis]))
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]

	# assemble the feature space for pre-drop timepoints
	predrop_coupling = weights_onto_neuron @ predrop_neural_data
	predrop_features = np.hstack((predrop_behavior, predrop_coupling[:, np.newaxis]))

	# assemble the feature space for post-drop timepoints
	postdrop_coupling = weights_onto_neuron @ postdrop_neural_data
	postdrop_features = np.hstack((postdrop_behavior, postdrop_coupling[:, np.newaxis]))

	if normalized:	
		# normalize the feature space by the maximum of the training data
		feat_maxes = np.max(feats[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = feats/feat_maxes
		predrop_features = predrop_features/feat_maxes
		postdrop_features = postdrop_features/feat_maxes

	if fit_params:
		# fit the alpha parameter (ridge regularization parameter) on the training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# otherwise use the number of trials/2
		alpha = features.shape[0]/16

	# train the regression
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate testing r^2 for the active or empty timepoints
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	# calculate r^2 for the pre-drop timepoints
	predrop_r2 = regression.score(predrop_features, predrop_neural_data[neuron, :])
	# calculate r^2 for the active or empty timepoints
	postdrop_r2 = regression.score(postdrop_features, postdrop_neural_data[neuron, :])
	
	if fig_save_dir is not None:
		# plot predicted vs data
		plt.plot(predrop_neural_data[neuron, :], label='data')
		plt.plot(regression.predict(predrop_features), label='prediction')
		plt.legend()
		plt.title(f'neuron {neuron}, r2={predrop_r2}')
		plt.savefig(fig_save_dir + f'predrop neuron {neuron} fold {which_fold}.png')
		plt.clf()

		plt.plot(postdrop_neural_data[neuron, :], label='data')
		plt.plot(regression.predict(postdrop_features), label='prediction')
		plt.legend()
		plt.title(f'neuron {neuron}, r2={postdrop_r2}')
		plt.savefig(fig_save_dir + f'postdrop neuron {neuron} fold {which_fold}.png')
		plt.clf()

	return [r2, predrop_r2, postdrop_r2]

def dim_control_ridge_regression_worker(neuron, which_fold, features, neural_data, weights_list, graph_bool, cv_splits, fit_params=False, normalized=False):
	'''
	worker function for parallel processing
	does one run of regression for a specific neuron and specific cv fold for FC-matched LOO models
	returns the test r^2 value on that fold for that neuron
	'''
	# grab training and testing splits for this CV fold for this neuron
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]
	# take the FN as the pre-calculated FN from the training data
	weights_onto_neuron = weights_list[neuron].copy()
	# remove the specified FCs (given for this replicate)
	weights_onto_neuron[graph_bool[neuron, :]]=0
	# calculate the coupling term as the dot product of the FN and neural data
	coupling_term = weights_onto_neuron @ neural_data
	# assemble the feature space
	feats = np.hstack((features, coupling_term[:, np.newaxis]))
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]
	
	if normalized:	
		# normalize features by the training data maxima
		feat_maxes = np.max(feats[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = feats/feat_maxes

	if fit_params:
		# fit ridge regularization parameter (alpha) on the training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# otherwise use alpha of the number of trials/2
		alpha = features.shape[0]/16

	# fit the regression on the training data
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate the testing r^2
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	return r2

def kin_only_regression_worker(neuron, which_fold, features, neural_data, cv_splits, fit_params=False, normalized=True):
	'''
	worker function for parallel processing
	does one run of regression for a specific neuron and specific cv fold for kinematics only models
	returns the test r^2 value on that fold for that neuron
	'''
	n_neurons = neural_data.shape[0]
	time = features.shape[0]

	# grab training and testing splits for this CV fold for this neuron
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]

	if normalized:	
		# normalize the features by the training data maxima
		feat_maxes = np.max(features[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = features/feat_maxes

	if fit_params:
		# fit ridge regularization parameter (alpha) on the training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# otherwise use an alpha of the number of trials/2
		alpha = features.shape[0]/16

	# fit the regression on the trainig data
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate the testing r^2
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	return r2

def weight_calculation_worker(neuron, cv_splits, which_fold, neural_data):
	'''
	worker function for parallel processing
	calculates and returns one row of a functional network for a given CV split on the training data
	'''
	# grab the training data
	training_idx = cv_splits[neuron][which_fold][0]
	# calculate the pearson correlation between a given neuron and all other neurons
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		corr_row = np.corrcoef(neural_data[:, training_idx])[neuron, :]
	# convert any NaNs to 0 (from if the data contains constants)
	weights_onto_neuron = np.nan_to_num(corr_row, nan=0.0, posinf=0.0, neginf=0.0)
	# set the neuron's correlation with itself (always 1) to 0 so that neuron data is not used to predict itself
	weights_onto_neuron[neuron] = 0
	return weights_onto_neuron

def parallelized_ridge_regression(features, neural_data, graph_bool, cross_cat_weights, cv_splits, n_cv_folds, normalized=True):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for matched- and mismatched-category kinematics and coupling models
	performs regressions parallelized across neurons, using all CPU cores
	'''

	n_neurons = neural_data.shape[0]
	# initialize arrays for testing r^2 scores
	r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	cross_cat_r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	
	for fold in range(n_cv_folds):
		print(f'working on fold {fold}...')
		# partial the worker to set everything except the neuron
		worker_func = functools.partial(
		within_cat_ridge_regression_worker,
		which_fold=fold, 
		features=features, 
		neural_data=neural_data, 
		cv_splits=cv_splits,
		graph_bool=graph_bool, 
		fit_params=True,
		normalized=normalized
		)
	
		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a list of r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		r2_scores[:, fold] = np.array(results)

		# partial the worker to set everything except the neuron
		worker_func = functools.partial(
		opposite_cat_ridge_regression_worker,
		which_fold=fold, 
		features=features, 
		neural_data=neural_data, 
		graph_like_weights=cross_cat_weights, 
		cv_splits=cv_splits,
		fit_params=True, 
		normalized=normalized
		)
	
		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a list of r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		cross_cat_r2_scores[:, fold] = np.array(results)

	return r2_scores, cross_cat_r2_scores

def parallelized_ridge_regression_no_weights(features, neural_data, cv_splits, n_cv_folds, normalized=True):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for a model using an FN where all non-diagonal points are 1
	performs regressions parallelized across neurons, using all CPU cores
	'''
	n_neurons = neural_data.shape[0]
	r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	# create a "FN" where every point in the matrix is 1
	weights = np.ones((n_neurons, n_neurons))
	# set all diagonals to 0 (so that neurons are not used to predict themselves)
	weights[np.arange(n_neurons), np.arange(n_neurons)] = 0
	
	for fold in range(n_cv_folds):
		print(f'working on fold {fold}...')
		# partial the worker to set everything except the neuron 
		worker_func = functools.partial(
		opposite_cat_ridge_regression_worker, # this worker allows for pre-calculated FNs
		which_fold=fold, 
		features=features, 
		neural_data=neural_data, 
		graph_like_weights=weights, 
		cv_splits=cv_splits,
		fit_params=True, 
		normalized=normalized
		)
	
		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a list of r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		r2_scores[:, fold] = np.array(results)

	return r2_scores

def parallelized_kinematics_only(features, neural_data, cv_splits, n_cv_folds, normalized=False):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for a kinematics-only model (no coupling term)
	performs regressions parallelized across neurons, using all CPU cores
	'''
	n_neurons = neural_data.shape[0]
	r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	
	for fold in range(n_cv_folds):
		print(f'working on fold {fold}...')
		# partial the worker to set everything except the neuron
		worker_func = functools.partial(
		kin_only_regression_worker,
		which_fold=fold, 
		features=features, 
		neural_data=neural_data, 
		cv_splits=cv_splits,
		fit_params=True,
		normalized=normalized
		)
	
		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a list of r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		r2_scores[:, fold] = np.array(results)

	return r2_scores

def permutation_ridge_regression_worker(permutation, which_fold, neuron, features, neural_data, weights_to_permute, graph_boolean, cv_splits, fit_params=False, normalized=True, nulls_permute=True):
	'''
	worker function for parallel processing
	does one run of regression for a specific neuron and specific cv fold for a FN permutation model
	returns the test r^2 value on that fold for that neuron
	'''
	n_neurons = neural_data.shape[0]
	# grab training and testing splits for this CV fold for this neuron
	training_idx, testing_idx = cv_splits[neuron][which_fold][0], cv_splits[neuron][which_fold][1]

	# create a boolean array stating which indices in the FN weights can be reassigned to
	all_but_these_boolean = np.ones(n_neurons, dtype=bool)
	# prevent reassignment of weights to diagonal indices
	all_but_these_boolean[neuron] = False
	if not nulls_permute: # if not allowing topological permutations
		# prevent reassignment of weights to null FCs
		all_but_these_boolean[~graph_boolean[neuron, :]] = False
	
	# randomly reassign all weights except those specified to remain as 0
	weights_onto_neuron = weights_to_permute.copy()
	weights_onto_neuron[all_but_these_boolean] = np.random.permutation(weights_onto_neuron[all_but_these_boolean])

	# calculate the coupling term as the dot product of the FN and neuronal activity
	coupling_term = weights_onto_neuron @ neural_data
	# assemble the feature space
	feats = np.hstack((features, coupling_term[:, np.newaxis]))
	# grab the predicted data
	neuron_Cascade = neural_data[neuron, :]
	
	if normalized:	
		# normalize the feature space by training data maxima
		feat_maxes = np.max(feats[training_idx, :], axis=0)
		# prevent dividing by 0
		if any(feat_maxes==0):
			zero_idx = np.where(feat_maxes==0)
			feat_maxes[zero_idx[0]]=1
		feats = feats/feat_maxes

	if fit_params:
		# fit ridge regularization parameter (alpha) on the training data
		model = Ridge()
		distributions = dict(alpha=np.linspace(0, features.shape[0], num=10))
		clf = GridSearchCV(model, distributions, scoring='r2')
		search = clf.fit(feats[training_idx, :], neuron_Cascade[training_idx])
		alpha=search.best_params_['alpha']
	else:
		# otherwise, use alpha of number of trials/2
		alpha = features.shape[0]/16

	# fit the regression on the training data
	regression = Ridge(alpha = alpha)
	regression.fit(feats[training_idx, :], neuron_Cascade[training_idx])
	# calculate the testing r^2
	r2 = regression.score(feats[testing_idx, :], neuron_Cascade[testing_idx])
	return r2

def parallelized_weight_permutation(features, neural_data, cv_splits, n_cv_folds, n_permutations, graph_bool, weights_edges_save_path = None, weights_save_path = None, normalized=True):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for weight only and weights and connectivity permutations, repeated n_permutations times
	performs regressions parallelized across neurons AND permutation repeats, using all CPU cores
	'''
	n_neurons = neural_data.shape[0]

	# initialize array to hold testing r^2 values for weights and connectivity permutations
	full_permutation_r2_values = np.zeros((n_neurons, n_cv_folds, n_permutations))*np.nan
	if weights_edges_save_path is not None: # if a save path is given
		if os.path.exists(weights_edges_save_path): # and exists
			full_permutation_r2_values = np.load(weights_edges_save_path) # load in the existing data

	# initialize array to hold testing r^2 values for weights-only permutations
	non_null_permutation_r2_values = np.zeros((n_neurons, n_cv_folds, n_permutations))*np.nan
	if weights_save_path is not None: # if a save path is given
		if os.path.exists(weights_save_path): # and exists
			non_null_permutation_r2_values = np.load(weights_save_path) # load in the existing data

	# create a dictionary with the weight calculation arguments except for neuron and fold
	state_wcalc = dict(features=features, neural_data=neural_data, cv_splits=cv_splits, normalized=normalized)
	for fold in range(n_cv_folds):
		# if a fold has already been completed, skip it
		if np.sum(np.isnan(full_permutation_r2_values[:, fold, :]))==0 and np.sum(np.isnan(non_null_permutation_r2_values[:, fold, :]))==0:
			print(f'already did fold {fold}, skipping...')
			continue
		print(f'working on fold {fold}...')
		
		# calculate FNs on the training data for this fold, returns as a list of the rows; paralellized across neurons
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_wcalc,)) as pool:
			weights_list = pool.map(_weight_calc_flat, [(neuron, fold) for neuron in range(n_neurons)])

		# zero-out the null FCs
		weights_per_neuron = [w.copy() for w in weights_list]
		for neuron in range(n_neurons):
			weights_per_neuron[neuron][~graph_bool[neuron, :]] = 0

		# create a dictionary with the permutation model arguments except for neuron, permutation_idx, and nulls_permute
		state_perm = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
		                  normalized=normalized, weights_per_neuron=weights_per_neuron,
		                  graph_bool=graph_bool, fit_params=True)
		
		# parallelize over neurons and permutation repeats
		# only open one multiprocessing pool to run both weights-only and weights and connectivity permutations
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_perm,)) as pool:
			# create tasks for all neurons, all permutation repeats with nulls_permute=True
			tasks_full = [(perm_idx, fold, neuron, True) for neuron in range(n_neurons) for perm_idx in range(n_permutations)]
			# map to a list of r^2 values with length n_neurons*n_permutations
			full_results = pool.map(_permutation_flat, tasks_full)
			
			# create tasks for all neurons, all permutation repeats with nulls_permute=False
			tasks_null = [(perm_idx, fold, neuron, False) for neuron in range(n_neurons) for perm_idx in range(n_permutations)]
			# map to a list of r^2 values with length n_neurons*n_permutations
			null_results = pool.map(_permutation_flat, tasks_null)
		
		# reshape into correct dimensions
		full_permutation_r2_values[:, fold, :] = np.array(full_results).reshape(n_neurons, n_permutations)
		non_null_permutation_r2_values[:, fold, :] = np.array(null_results).reshape(n_neurons, n_permutations)

		# save out after every fold
		if weights_edges_save_path is not None and weights_save_path is not None:
			np.save(weights_edges_save_path, full_permutation_r2_values)
			np.save(weights_save_path, non_null_permutation_r2_values)

	return full_permutation_r2_values, non_null_permutation_r2_values


def parallelized_LOO_models(features, neural_data, LOO_models, dim_control_models, cat_unique_graph_bool, other_cat_unique_graph_bool, shared_graph_bool, sign_flip_graph_bool, cv_splits, n_cv_folds, n_downsample_repeats, full_model_save_path=None, LOO_save_path=None, dim_control_save_path=None, threshold_r2=True, normalized=True, n_resamples=None):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for full model with null FCs NOT zeroed out; all LOO models, and all FC-count matched LOO models
	performs regressions parallelized:
		for the full model: across neurons, using all CPU cores
		for the full LOO models: across neurons AND FC types, using all CPU cores
		for the FC-count matched LOO models: across neurons AND FC types AND resamples, using all CPU cores
	'''
	if n_resamples is None:
		n_resamples = globals().get('n_resamples')
	n_neurons = neural_data.shape[0]

	# find the indices of each type of FC
	cat_unique_edge_indices = np.where(cat_unique_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	other_cat_unique_edge_indices = np.where(other_cat_unique_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	shared_edge_indices = np.where(shared_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	sign_flip_edge_indices = np.where(sign_flip_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	
	# count the number of FCs for each type of FC
	n_cat_unique_edges = len(cat_unique_edge_indices)
	n_other_cat_unique_edges = len(other_cat_unique_edge_indices)
	n_shared_edges = len(shared_edge_indices)
	n_sign_flip_edges = len(sign_flip_edge_indices)
	print(n_cat_unique_edges, n_other_cat_unique_edges, n_shared_edges, n_sign_flip_edges)

	if not os.path.exists(full_model_save_path): # if this has not been run and saved before
		# run the full model (all FCs in), parallelized across neurons
		print('running full model...')
		# initialize array to store r2 scores
		r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
		# "mask" becomes the graph boolean - e.g. what indices in the FN should not be converted to 0
		# here, it is everything (the within_cat function automatically zeroes out the diagonal)
		full_mask = np.ones((n_neurons, n_neurons), dtype=bool)
		# initialize the argument dictionary for the full model
		state_full = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
		                  normalized=normalized, masks=[full_mask])
		for fold in range(n_cv_folds):
			print(f'working on fold {fold}...')
			# parallelize over neurons
			with multiprocessing.Pool(initializer=_pool_init, initargs=(state_full,)) as pool:
				tasks = [(neuron, fold, 0) for neuron in range(n_neurons)]
				results = pool.map(_within_cat_flat, tasks)
			r2_scores[:, fold] = np.array(results)
		# save out the testing r^2 values after calculating them for all neurons, all folds
		np.save(full_model_save_path, r2_scores)
	else: # if the full model r^2 exist
		# load in the full model testing r^2 scores
		r2_scores = np.load(full_model_save_path)
	

	if threshold_r2:
		# calculate the mean and median r^2 across folds
		mean_r2_full = np.mean(r2_scores, axis=1)
		median_r2_full = np.median(r2_scores, axis=1)
		# find cells that have a cross-validated r^2 (either mean or median) greater than 0.05
		neuron_well_modelled = np.logical_or(mean_r2_full>=0.05, median_r2_full>=0.05)
		well_modelled_idx = np.where(neuron_well_modelled)[0]
		print(f'{np.sum(neuron_well_modelled)}/{n_neurons} well modelled')
	else:
		well_modelled_idx = np.arange(n_neurons)

	# run the LOO models
	print('running LOO models...')
	# initialize LOO testing r^2 array
	LOO_r2_values = np.zeros((n_neurons, n_cv_folds, len(LOO_models)))*np.nan
	if LOO_save_path is not None: # if a save path is given
		if os.path.exists(LOO_save_path): # and exists
			LOO_r2_values = np.load(LOO_save_path) # load in the already run r^2 values
	
	# create masks (graph_booleans) for each LOO model - this is what from the FN should be preserved
	# in this case, it is everything except the indices that are that FC type
	# this needs to be a list of graph booleans such that for multiprocessing model type it is iterated over
	loo_masks = []
	for model in LOO_models:
		if 'other_cat' in model:
			loo_masks.append(~other_cat_unique_graph_bool)
		elif 'category' in model:
			loo_masks.append(~cat_unique_graph_bool)
		elif 'share' in model:
			loo_masks.append(~shared_graph_bool)
		elif 'sign' in model:
			loo_masks.append(~sign_flip_graph_bool)
	# initialize all args except for neuron and fold
	state_loo = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
	                 normalized=normalized, masks=loo_masks)
	for fold in range(n_cv_folds):
		# skip any fold that has already been run
		if np.sum(np.isnan(LOO_r2_values)[well_modelled_idx, fold, :])==0:
			print(f'already did fold {fold}, skipping...')
			continue
		print(f'working on fold {fold}...')
		# parallelize over neurons and LOO model types (FC types)
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_loo,)) as pool:
			tasks = [(neuron, fold, model_i) for model_i in range(len(LOO_models)) for neuron in well_modelled_idx]
			# returns a list of length len(LOO_models)*len(well_modelled_idx)
			results = pool.map(_within_cat_flat, tasks)
		
		# convert list to array and reshape
		results_arr = np.array(results).reshape(len(LOO_models), len(well_modelled_idx))
		
		# assign to the correct indices in the full array
		for model_i in range(len(LOO_models)):
			LOO_r2_values[well_modelled_idx, fold, model_i] = results_arr[model_i, :]

		# save every 2 folds
		if (fold+1)%2 == 0:
			if LOO_save_path is not None:
				np.save(LOO_save_path, LOO_r2_values)

	# run the FC-count-matched LOO models
	print('running dimensionality control models...')
	# initialize an array for testing r^2 values
	dim_control_r2_values = np.zeros((n_neurons, n_cv_folds, len(dim_control_models)))*np.nan
	if dim_control_save_path is not None: # if a save path is given
		if os.path.exists(dim_control_save_path): # and exists
			dim_control_r2_values = np.load(dim_control_save_path) # load in the existing r^2 values

	# calculate the FNs for all folds (so that they don't need to be calculated on each resample)
	# initialize arguments for FN calculation, except neuron and fold
	state_wcalc = dict(features=features, neural_data=neural_data, cv_splits=cv_splits, normalized=normalized)
	for fold in range(n_cv_folds):
		# skip any fold that has already been entirely run
		if np.sum(np.isnan(dim_control_r2_values)[well_modelled_idx, fold, :])==0:
			print(f'already did fold {fold}, skipping...')
			continue
		print(f'working on fold {fold}...')
		# calculate FNs on the training data for this fold, returns as a list of the rows; paralellized across neurons
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_wcalc,)) as pool:
			weights_list = pool.map(_weight_calc_flat, [(neuron, fold) for neuron in range(n_neurons)])

		# presample all resampled FN matrices for all model types, make into a list to iterate over in the Pool
		all_masks = []
		for model_i, model in enumerate(dim_control_models):
			# grab the relevant indices and undersampling size for that FC-count matched model
			if model=='undersamp_cat_unique_to_shared':
				undersamp_edge_size = n_shared_edges
				sampling_indices = cat_unique_edge_indices.tolist()
			elif model=='undersamp_other_cat_unique_to_shared':
				undersamp_edge_size = n_shared_edges
				sampling_indices = other_cat_unique_edge_indices.tolist()
			elif model=='undersamp_cat_unique_to_sign_flip':
				undersamp_edge_size = n_sign_flip_edges
				sampling_indices = cat_unique_edge_indices.tolist()
			elif model=='undersamp_other_cat_unique_to_sign_flip':
				undersamp_edge_size = n_sign_flip_edges
				sampling_indices = other_cat_unique_edge_indices.tolist()
			elif model=='undersamp_shared_to_sign_flip':
				undersamp_edge_size = n_sign_flip_edges
				sampling_indices = shared_edge_indices.tolist()
			elif model=='undersamp_larger_cat_to_other_cat':
				if n_cat_unique_edges>n_other_cat_unique_edges:
					undersamp_edge_size=n_other_cat_unique_edges
					sampling_indices = cat_unique_edge_indices.tolist()
				else:
					undersamp_edge_size = n_cat_unique_edges
					sampling_indices = other_cat_unique_edge_indices.tolist()

			for resample in range(n_resamples): # for each resample
				# randomly grab indices of a given FC type from another FC type's count
				resampled_indices = random.sample(sampling_indices, undersamp_edge_size)
				# convert into a boolean matrix where only the indices to be REMOVED are true (diagonal are already removed)
				resampled_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
				resampled_edge_bool[resampled_indices] = True
				resampled_matrix_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
				resampled_matrix_bool[np.triu_indices(n_neurons, 1)] = resampled_edge_bool
				resampled_matrix_bool += resampled_matrix_bool.T
				all_masks.append(resampled_matrix_bool)

		# initialize arguments except neuron, resample, fold, and model
		state_dim = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
		                 normalized=normalized, masks=all_masks, weights_list=weights_list)
		# parallelize over neurons, models, and resamples
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_dim,)) as pool:
			tasks = [(neuron, fold, model_i * n_resamples + resample)
			         for model_i in range(len(dim_control_models))
			         for resample in range(n_resamples)
			         for neuron in well_modelled_idx]
			# result is a list of testing r^2 values of length len(dim_control_models)*n_resamples*len(well_modelled_idx)
			results = pool.map(_dim_control_flat, tasks)
		# convert to an array and rehape
		results_arr = np.array(results).reshape(len(dim_control_models), n_resamples, len(well_modelled_idx))
		
		# take the mean over resamples and assign to the proper index in the r^2 array
		for model_i in range(len(dim_control_models)):
			resampled_r2 = np.zeros((n_neurons, n_resamples))
			resampled_r2[well_modelled_idx, :] = results_arr[model_i, :, :].T
			# take the mean over resamples
			dim_control_r2_values[:, fold, model_i] = np.mean(resampled_r2, axis=1)

		# save after each fold
		if dim_control_save_path is not None:
			np.save(dim_control_save_path, dim_control_r2_values)

	return r2_scores, LOO_r2_values, dim_control_r2_values, well_modelled_idx

def drop_applied_ridge_regression(features, neural_data, graph_bool, predrop_neural, predrop_features, postdrop_neural, postdrop_features, cv_splits, n_cv_folds, fig_save_dir=None, normalized=True):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for the category it is trained on, pre-drop timepoints, and post-drop timepoints
	performs regressions parallelized across neurons, using all CPU cores
	'''
	n_neurons = neural_data.shape[0]
	# initialize arrays to save out r^2 values
	wi_cat_r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	predrop_r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	postdrop_r2_scores = np.zeros((n_neurons, n_cv_folds))*np.nan
	
	for fold in range(n_cv_folds):
		print(f'working on fold {fold}...')
		# partial the function to assign everything except for neuron
		worker_func = functools.partial(
		drop_applied_within_cat_ridge_regression_worker, 
		which_fold=fold,  
		features=features, 
		neural_data=neural_data, 
		graph_bool=graph_bool, 
		predrop_behavior=predrop_features, 
		predrop_neural_data=predrop_neural, 
		postdrop_behavior=postdrop_features, 
		postdrop_neural_data=postdrop_neural,
		cv_splits=cv_splits,
		fit_params=True,
		fig_save_dir=fig_save_dir,
		normalized=normalized
		)

		# Create a pool and map the work across cores
		with multiprocessing.Pool() as pool:
			# map distributes the n_neurons samples to the worker function
			# results is a (nested) list of tuples of within-cat, pre-drop, and post-drop r2 values with length n_neurons
			results = pool.map(worker_func, range(n_neurons))

		r2_scores = np.array(results) # shape: neurons x testing data type
		# assign r^2 scores to appropriate arrays
		wi_cat_r2_scores[:, fold] = r2_scores[:, 0]
		predrop_r2_scores[:, fold] = r2_scores[:, 1]
		postdrop_r2_scores[:, fold] = r2_scores[:, 2]

	return wi_cat_r2_scores, predrop_r2_scores, postdrop_r2_scores

def parallelized_edge_weight_LOO_models(features, neural_data, LOO_models, strong_weights_graph_bool, medium_weights_graph_bool, weak_weights_graph_bool, zero_weight_graph_bool, cv_splits, n_cv_folds, edge_proportions, n_resamples, well_modelled_idx, LOO_save_path=None, prop_LOO_save_path=None, normalized=True):
	'''
	calculates and returns the testing r^2 values for all folds, all neurons 
	for all LOO models and all FC-count matched LOO models
	performs regressions parallelized:
		for the full LOO models: across neurons AND FC weight strength levels, using all CPU cores
		for the FC-count matched LOO models: across neurons AND FC weight strength levels AND proportion removed AND resamples, using all CPU cores
	'''
	n_neurons = neural_data.shape[0]

	# find the indices of each type of FC by weight magnitude
	strong_edge_indices = np.where(strong_weights_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	medium_edge_indices = np.where(medium_weights_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	weak_edge_indices = np.where(weak_weights_graph_bool[np.triu_indices(n_neurons, 1)])[0]
	
	# count each FC type
	n_strong_edges = len(strong_edge_indices)
	n_medium_edges = len(medium_edge_indices)
	n_weak_edges = len(weak_edge_indices)
	print(n_strong_edges, n_medium_edges, n_weak_edges)

	# run LOO models (removing all FCs of a given weight magnitude level)
	print('running LOO models...')
	# initialize array for testing r^2 values
	LOO_r2_values = np.zeros((n_neurons, n_cv_folds, len(LOO_models)))*np.nan
	if LOO_save_path is not None: # if a save path is given
		if os.path.exists(LOO_save_path): # and exists
			LOO_r2_values = np.load(LOO_save_path) # load in the existing data
	
	# create masks (graph_booleans) for which FN indices should be PRESERVED for each LOO model
	loo_masks = []
	for model in LOO_models:
		if 'strong' in model:
			loo_masks.append(~strong_weights_graph_bool)
		elif 'med' in model:
			loo_masks.append(~medium_weights_graph_bool)
		elif 'weak' in model:
			loo_masks.append(~weak_weights_graph_bool)
		elif 'zero' in model:
			loo_masks.append(~zero_weight_graph_bool)
	
	# initialize arguments except for neuron, fold, and model
	state_loo = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
	                 normalized=normalized, masks=loo_masks)
	for fold in range(n_cv_folds):
		# if the fold has already been run, skip it
		if np.sum(np.isnan(LOO_r2_values[well_modelled_idx, fold, :]))==0:
			print(f'already did fold {fold}, skipping...')
			continue
		print(f'working on fold {fold}...')
		# parallelize over LOO models (FC weight magnitude classes) and neurons
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_loo,)) as pool:
			tasks = [(neuron, fold, model_i) for model_i in range(len(LOO_models)) for neuron in well_modelled_idx]
			# results is a list of tesing r^2 values of length len(LOO_models)*len(well_modelled_idx)
			results = pool.map(_within_cat_flat, tasks)
		# convert to an array and reshape
		results_arr = np.array(results).reshape(len(LOO_models), len(well_modelled_idx))
		# assign to the appropriate locations in the larger array
		for model_i in range(len(LOO_models)):
			LOO_r2_values[well_modelled_idx, fold, model_i] = results_arr[model_i, :]
	# save the r^2 values
	if LOO_save_path is not None:
		np.save(LOO_save_path, LOO_r2_values)

	# run the count-matched LOO models
	print('running dimensionality control models...')
	# initialize an array for testing r^2 values
	dim_control_r2_values = np.zeros((n_neurons, n_cv_folds, len(edge_proportions), len(LOO_models[:-1])))*np.nan
	if prop_LOO_save_path is not None: # if a save path is given
		if os.path.exists(prop_LOO_save_path): # and exists
			dim_control_r2_values = np.load(prop_LOO_save_path) # load in the existing data

	# calculate the FNs for all folds (so that they don't need to be calculated on each resample)
	# initialize arguments for FN calculation, except neuron and fold
	state_wcalc = dict(features=features, neural_data=neural_data, cv_splits=cv_splits, normalized=normalized)
	for fold in range(n_cv_folds):
		# skip any fold that has been entirely run already
		if np.sum(np.isnan(dim_control_r2_values[well_modelled_idx, fold, :, :]))==0:
			print(f'already did fold {fold}, skipping...')
			continue
		print(f'working on fold {fold}...')
		# calculate FNs on the training data for this fold, returns as a list of the rows; paralellized across neurons
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_wcalc,)) as pool:
			weights_list = pool.map(_weight_calc_flat, [(neuron, fold) for neuron in range(n_neurons)])

		# presample all resampled FN matrices for all model types and proportion removed
		# make into a list to iterate over in the Pool
		all_masks = []
		for prop_i, proportion in enumerate(edge_proportions):
			# get the count of FCs to remove for each proportion removed
			# all proportions are less than that of a single FC weight magnitude class
			undersamp_edge_size = int(proportion*np.sum(~zero_weight_graph_bool[np.triu_indices(n_neurons, 1)]))
			# grab the relevant indices for each LOO model
			for model_i, model in enumerate(LOO_models[:-1]):
				if 'strong' in model:
					sampling_indices = strong_edge_indices.tolist()
				elif 'med' in model:
					sampling_indices = medium_edge_indices.tolist()
				elif 'weak' in model:
					sampling_indices = weak_edge_indices.tolist()

				for resample in range(n_resamples): # for each resampled
					# grab a random set of indices of the given FC weight magntiude type at the undersampling size
					resampled_indices = random.sample(sampling_indices, undersamp_edge_size)
					# convert into a boolean matrix where only the indices to be REMOVED are true (diagonal are already removed)
					resampled_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)[np.triu_indices(n_neurons, 1)]
					resampled_edge_bool[resampled_indices] = True
					resampled_matrix_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
					resampled_matrix_bool[np.triu_indices(n_neurons, 1)] = resampled_edge_bool
					resampled_matrix_bool += resampled_matrix_bool.T
					#resampled_matrix_bool[zero_weight_graph_bool] = True # can only keep if comparing to model without null edges
					all_masks.append(resampled_matrix_bool)

		# initialize arguments for FC-count matched LOO models except neuron, fold
		state_dim = dict(features=features, neural_data=neural_data, cv_splits=cv_splits,
		                 normalized=normalized, masks=all_masks, weights_list=weights_list)
		n_outer_models = len(LOO_models[:-1]) # not running zero
		# parallelize over neurons, proportions removed, models, and resamples
		with multiprocessing.Pool(initializer=_pool_init, initargs=(state_dim,)) as pool:
			# this sets up to iterate over the list of masks (graph booleans of FCs to remove)
			tasks = [(neuron, fold, ((prop_i * n_outer_models + model_i) * n_resamples + resample))
			         for prop_i in range(len(edge_proportions))
			         for model_i in range(n_outer_models)
			         for resample in range(n_resamples)
			         for neuron in well_modelled_idx]
			# results is a list of test r^2 values of length len(edge_proportions)*n_outer_models*n_resamples*len(well_modelled_idx)
			results = pool.map(_dim_control_flat, tasks)
		# convert to an array and reshape
		results_arr = np.array(results).reshape(len(edge_proportions), n_outer_models, n_resamples, len(well_modelled_idx))
		
		# assign to the proper indexes in the larger r^2 array
		for prop_i in range(len(edge_proportions)):
			for model_i in range(n_outer_models):
				resampled_r2 = np.zeros((n_neurons, n_resamples))
				resampled_r2[well_modelled_idx, :] = results_arr[prop_i, model_i, :, :].T
				# take the mean over resamples
				dim_control_r2_values[well_modelled_idx, fold, prop_i, model_i] = np.mean(resampled_r2[well_modelled_idx, :], axis=1)

		# save out every fold
		if prop_LOO_save_path is not None:
			np.save(prop_LOO_save_path, dim_control_r2_values)

	return LOO_r2_values, dim_control_r2_values


mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

normalized=True

# what in this script to run
calculate_splits=True
kinematic_only = True
nulled_coupling = True
apply_to_drops = True
unique_LOOs = True
edge_strength_LOOs=True
no_weights = True
permute_edges = True

t_pre = 4 # neural frames relative to rotation time
t_post = 4 # neural frames relative to rotation time
time_in_trial = t_pre + t_post # neural frames
lag=int(2) # frames

if calculate_new_splits:
	for splitter in ['Shuffle', 'KFold']:
		if splitter=='Shuffle':
			cv_folds=10
			test_size=0.3
		elif splitter=='KFold':
			cv_folds=2
			n_repeats=10
		for mouseID in mice:
			days = src.IO.get_carry_days(mouseID)
			drive = '/home/macleanlab/elizawiener/data/'
			mouse_dir = drive + '/' + mouseID + '/'
			save_dir = mouse_dir + 'carry_analysis/GLM/'
			reg_inds = src.utils.load_registered_cells(mouse_dir, days)

			neural_activity_by_trial = [[] for day in days]
			time_cat_labels = [[] for day in days]
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

				# get trial-specific neural data
				temp_spks = [[] for time in carry_times]
				for carry_i, carry in enumerate(carry_times.astype(int)):
					temp_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
					trial_labels[i].extend(np.ones(time_in_trial)*trial_counter)
					trial_counter+=1
				neural_activity_by_trial[i] = np.hstack(temp_spks)
				day_labels[i] = np.ones(len(category_labels_by_time))*i
				trial_like_day_labels[i] = np.ones(len(carry_times))*i
				time_cat_labels[i] = category_labels_by_time

			# combine across days
			neural_activity = np.hstack(neural_activity_by_trial)
			day_labels = np.hstack(day_labels)
			trial_labels = np.hstack(trial_labels)
			trial_like_day_labels = np.hstack(trial_like_day_labels)
			trial_like_cat_labels = np.hstack(trial_like_cat_labels)
			time_cat_labels = np.hstack(time_cat_labels)
			print(day_labels.shape, trial_labels.shape, time_cat_labels.shape)

			# remove neurons with only NaN for activity
			activity_nonan = neural_activity[~np.isnan(neural_activity).any(axis=1), :]
			print(np.sum(np.isnan(neural_activity).any(axis=1)), ' NaN-containing cells removed')
			NaN_indices = np.where(np.isnan(neural_activity).any(axis=1))[0]
			red_cells_bool = red_cells[~np.isnan(neural_activity).any(axis=1)]

			n_neurons = activity_nonan.shape[0]
			print('n cells: ', n_neurons)

			# loop over categories
			for category in ['active', 'empty']:
				if category=='active':
					label=0
				else:
					label=1

				category_idx = time_cat_labels==label
				trial_like_cat_idx = trial_like_cat_labels==label
				cat_day_labels = day_labels[category_idx]
				cat_neural_data = activity_nonan[:, category_idx]
				cat_trial_labels = trial_labels[category_idx]
				
				# first split trial-wise
				if splitter=='Shuffle':
					cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size)
					cv_splits = [[[[], []] for fold in range(cv_folds)] for neuron in range(n_neurons)]
					cv_split_indices = [[[[], []] for fold in range(cv_folds)] for neuron in range(n_neurons)]
				elif splitter=='KFold':
					cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=n_repeats)
					cv_splits = [[[[], []] for fold in range(cv_folds*n_repeats)] for neuron in range(n_neurons)]
					cv_split_indices = [[[[], []] for fold in range(cv_folds*n_repeats)] for neuron in range(n_neurons)]
				
				# for each neuron, find the average firing rate in each trial
				for neuron in range(n_neurons):
					trial_avg_frs = np.zeros(len(np.unique(cat_trial_labels)))*np.nan
					for trial_i, trial in enumerate(np.unique(cat_trial_labels)):
						trial_avg_frs[trial_i] = np.mean(cat_neural_data[neuron, cat_trial_labels==trial])
					# create bins of trial average firing rate
					_, bin_edges = np.histogram(trial_avg_frs, bins=5)
					
					# categorize each trial into the bins
					binned_trial_frs = np.zeros(len(trial_avg_frs))*np.nan
					for trial_i, trial_fr in enumerate(trial_avg_frs):
						if trial_fr<bin_edges[1]:
							binned_trial_frs[trial_i]=0
						elif trial_fr<bin_edges[2]:
							binned_trial_frs[trial_i]=1
						elif trial_fr<bin_edges[3]:
							binned_trial_frs[trial_i]=2
						elif trial_fr<bin_edges[4]:
							binned_trial_frs[trial_i]=3
						elif trial_fr<=bin_edges[5]:
							binned_trial_frs[trial_i]=4

					# create stratification labels for trials based on firing rate bin
					labels_for_strat = binned_trial_frs

					# if there is any stratification label that only exists for one trial, exclude that trial
					# count labels
					c = Counter(labels_for_strat)
					trials_to_include = np.ones(len(labels_for_strat), dtype=bool)
					for label in np.unique(labels_for_strat):
						if splitter=='Shuffle':
							if c[label]<2:
								trials_to_include[labels_for_strat==label] = False
						elif splitter=='KFold':
							if c[label]<cv_folds:
								trials_to_include[labels_for_strat==label] = False
					print(f'{np.sum(~trials_to_include)}/{len(trials_to_include)} trials excluded')

					# remove trials that have been excluded for really weird firing rates
					labels = labels_for_strat[trials_to_include]
					included_trial_labels = np.unique(cat_trial_labels)[trials_to_include]

					if splitter=='Shuffle':
						# make cv fold splits and save them
						cv_splits[neuron] = cv.split(included_trial_labels, labels)
						# convert to timepoint-based train/test indices
						for fold, (train, test) in enumerate(cv_splits[neuron]):
							training_idx = np.concatenate([np.where(cat_trial_labels==trial)[0] for trial in included_trial_labels[train]])
							testing_idx = np.concatenate([np.where(cat_trial_labels==trial)[0] for trial in included_trial_labels[test]])
							cv_split_indices[neuron][fold][0], cv_split_indices[neuron][fold][1] = training_idx, testing_idx
						# save the nested list of splits as a pickle
						src.IO.save_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl', cv_split_indices)
					elif splitter=='KFold':
						# make cv fold splits and save them
						cv_splits[neuron] = cv.split(included_trial_labels, labels)
						# convert to timepoint-based train/test splits
						for fold, (train, test) in enumerate(cv_splits[neuron]):
							training_idx = np.concatenate([np.where(cat_trial_labels==trial)[0] for trial in included_trial_labels[train]])
							testing_idx = np.concatenate([np.where(cat_trial_labels==trial)[0] for trial in included_trial_labels[test]])
							cv_split_indices[neuron][fold][0], cv_split_indices[neuron][fold][1] = training_idx, testing_idx
						# save the nested list of splits as a pickle
						src.IO.save_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl', cv_split_indices)

				
				# then split timepoint-wise
				if splitter=='KFold':
					cv = StratifiedKFold(n_splits=cv_folds)
				elif splitter=='Shuffle':
					cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=0.3)
				cv_splits = [[[[], []] for fold in range(cv_folds)] for neuron in range(n_neurons)]
				cv_split_indices = [[[[], []] for fold in range(cv_folds)] for neuron in range(n_neurons)]
				# for each neuron, find the average firing rate in each trial
				for neuron in range(n_neurons):
					# create bins of instant firing rate
					_, bin_edges = np.histogram(cat_neural_data[neuron, :], bins=3)
					binned_timepoints = np.zeros(len(cat_neural_data[neuron, :]))*np.nan
					for t_i, fr in enumerate(cat_neural_data[neuron, :]):
						if fr<bin_edges[1]:
							binned_timepoints[t_i]=0
						elif fr<bin_edges[2]:
							binned_timepoints[t_i]=1
						elif fr<=bin_edges[3]:
							binned_timepoints[t_i]=2
		
					# create stratification labels for trials based on day and firing rate bin
					labels_for_strat = np.array([f"{a}_{b}" for a,b in zip(cat_day_labels, binned_timepoints)])
					
					if splitter=='Shuffle':
						# if using stratified shuffle split, need >=2 members for each label
						trials_to_include = np.ones(len(labels_for_strat), dtype=bool)
						c = Counter(labels_for_strat)
						for label in np.unique(labels_for_strat):
							if c[label]<2:
								trials_to_include[labels_for_strat==label] = False

						# make cv fold splits
						cv_splits[neuron] = cv.split(cat_features[trials_to_include, :], labels_for_strat[trials_to_include])
					
					else:
						# make cv fold splits
						cv_splits[neuron] = cv.split(cat_features, labels_for_strat)

					# convert to timepoint-based train/test indices
					for fold, (train, test) in enumerate(cv_splits[neuron]):
						cv_split_indices[neuron][fold][0], cv_split_indices[neuron][fold][1] = train, test
				
				# save the splits as a pickled nested list
				src.IO.save_pickle(save_dir + f'{category}_timepoint_{splitter}_cv_splits_{cv_folds}_folds.pkl', cv_split_indices)

if kinematic_only:
	# set the cross-validation paramters
	split_method=='trial'
	splitter = 'KFold' 
	cv_folds=2
	n_repeats=10
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'

		# load in cross-day cell registration
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
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
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			if category=='active':
				label=0
			else:
				label=1

			category_idx = behavior[where_not_NaN, -1]==label
			trial_like_cat_idx = trial_like_cat_labels==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			cat_trial_labels = trial_labels[category_idx]
			
			# import cv split indices
			if splitter=='Shuffle':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
			elif splitter=='KFold':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')				

			# run models
			if normalized:
				if splitter=='Shuffle':
					r2_values = parallelized_kinematics_only(cat_features, cat_neural_data, cv_split_indices, cv_folds, normalized=True)
				elif splitter=='KFold':
					r2_values = parallelized_kinematics_only(cat_features, cat_neural_data, cv_split_indices, cv_folds*n_repeats, normalized=True)
			else:
				if splitter=='Shuffle':
					r2_values = parallelized_kinematics_only(cat_features, cat_neural_data, cv_split_indices, cv_folds)
				elif splitter=='KFold':
					r2_values = parallelized_kinematics_only(cat_features, cat_neural_data, cv_split_indices, cv_folds*n_repeats)
			
			# print out results
			print(mouseID)
			print(category)
			print('median full model r2: ', np.median(np.mean(r2_values, axis=1)))
			print()

			# save results by category, lag, and cv split type
			if normalized:
				if splitter=='Shuffle':
					np.save(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{test_size}split_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', r2_values)
				elif splitter=='KFold':
					np.save(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', r2_values)
			else:
				if splitter=='Shuffle':
					np.save(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{test_size}split_ridge_regression_r2_values_distr_strat.npy', r2_values)
				elif splitter=='KFold':
					np.save(save_dir + f'{category}_kinematics_only_lag{lag}_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy', r2_values)

if nulled_coupling:
	# set the cross-validation specifications
	split_method=='trial'
	splitter = 'KFold'
	cv_folds=2
	n_repeats=10
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
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
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)

		# load in null/non-null FCs for each FN
		sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl')
		# make a boolean matrix indicating where FN is null/non-null for each category
		active_edge_bool = np.zeros(len(np.zeros((n_neurons, n_neurons))[np.triu_indices(n_neurons, 1)]), dtype=bool)
		active_edge_bool[sig_weight_indices['active']]=True
		empty_edge_bool = np.zeros(len(active_edge_bool), dtype=bool)
		empty_edge_bool[sig_weight_indices['empty']]=True

		# import weights for active and empty
		active_graph = np.load(s2p_fld + f'active_FN_pearson_corr.npy')
		empty_graph = np.load(s2p_fld + f'empty_FN_pearson_corr.npy')

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			if category=='active':
				label=0
				weight_bool=active_edge_bool
				cross_cat_weights = empty_graph
			else:
				label=1
				weight_bool=empty_edge_bool
				cross_cat_weights=active_graph

			# convert weight boolean array into graph-like boolean matrix
			graph_like_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_edge_bool[np.triu_indices(n_neurons, 1)] = weight_bool
			graph_like_edge_bool += graph_like_edge_bool.T

			# zero out category-specific nulls in the cross-category FN
			cross_cat_weights[~graph_like_edge_bool]=0
			cross_cat_weights[np.diag_indices(n_neurons)]=0

			# grab trials, features, day labels, and neural data for the given category
			category_idx = behavior[where_not_NaN, -1]==label
			trial_like_cat_idx = trial_like_cat_labels==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			cat_trial_labels = trial_labels[category_idx]
			
			# import cv split indices
			if splitter=='Shuffle':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
			elif splitter=='KFold':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')				

			# run regression models
			if normalized:
				if splitter=='Shuffle':
					r2_values, cross_cat_r2_values = parallelized_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, cross_cat_weights, cv_split_indices, cv_folds, normalized=True)
				elif splitter=='KFold':
					r2_values, cross_cat_r2_values = parallelized_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, cross_cat_weights, cv_split_indices, cv_folds*n_repeats, normalized=True)
			else:
				if splitter=='Shuffle':
					r2_values, cross_cat_r2_values = parallelized_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, cross_cat_weights, cv_split_indices, cv_folds)
				elif splitter=='KFold':
					r2_values, cross_cat_r2_values = parallelized_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, cross_cat_weights, cv_split_indices, cv_folds*n_repeats)
			
			# print matched and mismatched-category results
			print(mouseID)
			print(category)
			print('median matched-category coupling r2: ', np.median(np.mean(r2_values, axis=1)))
			print('median mismatched-category coupling r2: ', np.median(np.mean(cross_cat_r2_values, axis=1)))
			print('median r2 difference: ', np.median(np.mean(cross_cat_r2_values-r2_values, axis=1)))
			print()

			# save out r2 scores
			if normalized:
				if splitter=='Shuffle':
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', cross_cat_r2_values)
				elif splitter=='KFold':
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy', cross_cat_r2_values)
			else:
				if splitter=='Shuffle':
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{test_size}split_ridge_regression_r2_values_distr_strat.npy', cross_cat_r2_values)
				elif splitter=='KFold':
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy', cross_cat_r2_value)

if unique_LOOs:
	# CV split specifications
	split_method=='trial'
	splitter = 'KFold'
	cv_folds=2
	n_repeats=10

	# model types 
	LOO_models = ['category_unique_edges', 'other_cat_unique_edges', 'shared_edges', 'sign_flip']
	dim_control_models = ['undersamp_cat_unique_to_shared', 'undersamp_other_cat_unique_to_shared', 'undersamp_cat_unique_to_sign_flip', 
		'undersamp_other_cat_unique_to_sign_flip', 'undersamp_shared_to_sign_flip', 'undersamp_larger_cat_to_other_cat']
	n_resamples = 500 # for dimensionality control models
	
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
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
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)

		# load in unique edges
		edges = src.IO.load_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_weighted_PDF_10000_pearson_corr.pkl')
		unique_shared = {'unique_active':[], 'unique_empty':[], 'shared':[], 'sign_flip':[], 'both_zero':[]}
		print('n unique active edges: ', len(edges['unique_active']))
		print('n unique empty edges: ', len(edges['unique_empty']))
		print('n shared edges: ', len(edges['shared']))
		print('n sign flip edges: ', len(edges['sign_flip']))

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

		print(np.sum(sign_flip_edge_bool))
		print(np.sum(graph_like_sign_flip_edge_bool[np.triu_indices(n_neurons, 1)]))

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			print(category)
			if category=='active':
				label=0
				cross_cat_unique_edge_idx = edges['unique_empty']
			else:
				label=1
				cross_cat_unique_edge_idx = edges['unique_active']

			# index unique edges
			unique_edge_idx = edges[f'unique_{category}']
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

			# grab category-specific data
			category_idx = behavior[where_not_NaN, -1]==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			
			if splitter=='Shuffle':
				# import cv split indices
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
				# run models
				if normalized:
					r2_values, LOO_r2s, dim_control_r2s, well_modelled = parallelized_LOO_models(cat_features, cat_neural_data, LOO_models, dim_control_models, graph_like_unique_edge_bool, graph_like_cross_cat_unique_edge_bool, graph_like_shared_edge_bool, graph_like_sign_flip_edge_bool, cv_split_indices, cv_folds, n_resamples, normalized=True)
				else:
					r2_values, LOO_r2s, dim_control_r2s, well_modelled = parallelized_LOO_models(cat_features, cat_neural_data, LOO_models, dim_control_models, graph_like_unique_edge_bool, graph_like_cross_cat_unique_edge_bool, graph_like_shared_edge_bool, graph_like_sign_flip_edge_bool, cv_split_indices, cv_folds, n_resamples)
			elif splitter=='KFold':
				# import cv split indices
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
				# run models
				if normalized:
					r2_values, LOO_r2s, dim_control_r2s, well_modelled = parallelized_LOO_models(cat_features, cat_neural_data, LOO_models, dim_control_models, graph_like_unique_edge_bool, graph_like_cross_cat_unique_edge_bool, graph_like_shared_edge_bool, graph_like_sign_flip_edge_bool, cv_split_indices, cv_folds*n_repeats, n_resamples, normalized=True)
				else:
					r2_values, LOO_r2s, dim_control_r2s, well_modelled = parallelized_LOO_models(cat_features, cat_neural_data, LOO_models, dim_control_models, graph_like_unique_edge_bool, graph_like_cross_cat_unique_edge_bool, graph_like_shared_edge_bool, graph_like_sign_flip_edge_bool, cv_split_indices, cv_folds*n_repeats, n_resamples)

			# print out results
			print(mouseID)
			print(category)
			print('median full model r2: ', np.median(np.mean(r2_values, axis=1)))
			for model_i, LOO_model in enumerate(LOO_models):
				print(f'median left out {LOO_model} delta r2: ', np.median(np.mean(LOO_r2s[well_modelled, :, model_i]-r2_values[well_modelled, :], axis=1)))
			for model_i, LOO_model in enumerate(dim_control_models):
				print(f'median left out {LOO_model} delta r2: ', np.median(np.mean(dim_control_r2s[well_modelled, :, model_i]-r2_values[well_modelled, :], axis=1)))
			print()

			# save r2 values
			if normalized:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges_NORMALIZED.npy', r2_values)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_NORMALIZED.npy', LOO_r2s)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control_NORMALIZED.npy', dim_control_r2s)
			else:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges.npy', r2_values)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness.npy', LOO_r2s)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control.npy', dim_control_r2s)


if edge_strength_LOOs:
	# CV split specifications
	split_method=='trial'
	splitter = 'KFold' 
	cv_folds=2
	n_repeats=10

	# LOO model types
	LOO_models = ['strong', 'medium', 'weak', 'zero']
	removed_edge_proportions = [.01, .05, .1]
	strong_weights_percentile = 90 # top 10% of weights
	weak_weights_percentile = 25 # bottom 25% of weights

	n_resamples = 100
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_labels[i].extend(np.ones(time_in_trial)*trial_counter)
				trial_counter+=1
			neural_activity_by_trial[i] = np.hstack(temp_spks)
			behavior[i] = np.hstack(temp_behavior)
			behavior[i] = np.vstack((behavior[i], category_labels_by_time))
			day_labels[i] = np.ones(behavior[i].shape[1])*i
			trial_like_day_labels[i] = np.ones(len(carry_times))*i

		# combine data across days
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
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)

		# load in significant edge indices
		sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl')
		active_edge_bool = np.zeros(len(np.zeros((n_neurons, n_neurons))[np.triu_indices(n_neurons, 1)]), dtype=bool)
		active_edge_bool[sig_weight_indices['active']]=True
		empty_edge_bool = np.zeros(len(active_edge_bool), dtype=bool)
		empty_edge_bool[sig_weight_indices['empty']]=True

		# load in a priori calculated FNs
		active_graph = np.load(s2p_fld + f'active_FN_pearson_corr.npy')
		empty_graph = np.load(s2p_fld + f'empty_FN_pearson_corr.npy')

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			if category=='active':
				label=0
				weight_bool=active_edge_bool
				weights = active_graph
			else:
				label=1
				weight_bool=empty_edge_bool
				weights = empty_graph

			# find strong, weak, and medium weight magnitude cutoffs
			strong_weight_threshold = np.percentile(np.abs(weights[np.triu_indices(n_neurons, 1)][weight_bool]), strong_weights_percentile)
			weak_weight_threshold = np.percentile(np.abs(weights[np.triu_indices(n_neurons, 1)][weight_bool]), weak_weights_percentile)

			print(strong_weight_threshold)
			print(weak_weight_threshold)

			# make boolean arrays using the thresholds to specify strong, medium and weak FCs
			strong_edge_bool = np.abs(weights[np.triu_indices(n_neurons, 1)])>=strong_weight_threshold
			weak_edge_bool = np.abs(weights[np.triu_indices(n_neurons, 1)])<weak_weight_threshold
			weak_edge_bool[~weight_bool]=False
			medium_edge_bool = np.logical_and(np.abs(weights[np.triu_indices(n_neurons, 1)])>=weak_weight_threshold, np.abs(weights[np.triu_indices(n_neurons, 1)])<strong_weight_threshold)
			medium_edge_bool[~weight_bool]=False

			# convert into boolean matricies
			graph_like_strong_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_strong_edge_bool[np.triu_indices(n_neurons, 1)] = strong_edge_bool
			graph_like_strong_edge_bool += graph_like_strong_edge_bool.T

			graph_like_weak_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_weak_edge_bool[np.triu_indices(n_neurons, 1)] = weak_edge_bool
			graph_like_weak_edge_bool += graph_like_weak_edge_bool.T

			graph_like_medium_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_medium_edge_bool[np.triu_indices(n_neurons, 1)] = medium_edge_bool
			graph_like_medium_edge_bool += graph_like_medium_edge_bool.T

			graph_like_zero_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_zero_edge_bool[np.triu_indices(n_neurons, 1)] = ~weight_bool
			graph_like_zero_edge_bool += graph_like_zero_edge_bool.T

			# load in LOO models to get indices of well modeled neurons
			well_modelled = ~np.isnan(np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness.npy')[:, 0, 0])
			print(np.sum(well_modelled))
			well_modelled_idx = np.where(well_modelled)[0]

			# get category-specific data
			category_idx = behavior[where_not_NaN, -1]==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			
			# import cv split indices
			if splitter=='Shuffle':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
				n_folds_total = cv_folds
			elif splitter=='KFold':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
				n_folds_total = cv_folds*n_repeats

			# run models
			if normalized:
				LOO_r2s, prop_LOO_r2s = parallelized_edge_weight_LOO_models(cat_features, cat_neural_data, LOO_models, graph_like_strong_edge_bool, graph_like_medium_edge_bool, 
					graph_like_weak_edge_bool, graph_like_zero_edge_bool, cv_split_indices, n_folds_total, removed_edge_proportions, n_resamples, well_modelled_idx, normalized=True)
			else:
				LOO_r2s, prop_LOO_r2s = parallelized_edge_weight_LOO_models(cat_features, cat_neural_data, LOO_models, graph_like_strong_edge_bool, graph_like_medium_edge_bool, 
					graph_like_weak_edge_bool, graph_like_zero_edge_bool, cv_split_indices, n_folds_total, removed_edge_proportions, n_resamples, well_modelled_idx)
			
			# load in all FCs model to compare to
			r2_values = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges.npy')
			
			# print strong/medium/weak thresholds
			print(mouseID)
			print(category)
			print('strong: ', strong_weight_threshold, '-', np.max(np.abs(weights[np.triu_indices(n_neurons, 1)][weight_bool])))
			print('weak: ', weak_weight_threshold, '-', np.min(np.abs(weights[np.triu_indices(n_neurons, 1)][weight_bool])))
			print('medium: ', weak_weight_threshold, '-', strong_weight_threshold)

			# save r2 values
			if normalized:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_NORMALIZED.npy', LOO_r2s)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_by_proportion_of_edges_{removed_edge_proportions}_NORMALIZED.npy', prop_LOO_r2s)
			else:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength.npy', LOO_r2s)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_by_proportion_of_edges_{removed_edge_proportions}.npy', prop_LOO_r2s)

			# print delta r2 compared to full model
			print('median full model r2: ', np.median(np.mean(r2_values[well_modelled, :], axis=1)))
			for model_i, LOO_model in enumerate(LOO_models):
				print(f'median left out {LOO_model} delta r2: ', np.median(np.mean(LOO_r2s[well_modelled, :, model_i]-r2_values[well_modelled, :], axis=1)))
			for prop_i, proportion in enumerate(removed_edge_proportions):
				print(proportion, ' edges removed: ')
				for model_i, LOO_model in enumerate(LOO_models[:-1]):
					print(f'median left out {LOO_model} delta r2: ', np.median(np.mean(prop_LOO_r2s[well_modelled, :, prop_i, model_i]-r2_values[well_modelled, :], axis=1)))
			print()

if no_weights:
	# specify cv split information
	split_method=='trial'
	splitter = 'KFold'
	cv_folds=2
	n_repeats=10

	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_labels[i].extend(np.ones(time_in_trial)*trial_counter)
				trial_counter+=1
			neural_activity_by_trial[i] = np.hstack(temp_spks)
			behavior[i] = np.hstack(temp_behavior)
			behavior[i] = np.vstack((behavior[i], category_labels_by_time))
			day_labels[i] = np.ones(behavior[i].shape[1])*i
			trial_like_day_labels[i] = np.ones(len(carry_times))*i

		# combine data across days
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
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			if category=='active':
				label=0
			else:
				label=1

			# grab category-specific data
			category_idx = behavior[where_not_NaN, -1]==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			
			# import cv split indices
			if splitter=='Shuffle':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
				n_folds_total = cv_folds
			elif splitter=='KFold':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
				n_folds_total = cv_folds*n_repeats

			# run models
			if normalized:
				r2 = parallelized_ridge_regression_no_weights(cat_features, cat_neural_data, cv_split_indices, n_folds_total, normalized=True)
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				r2 = parallelized_ridge_regression_no_weights(cat_features, cat_neural_data, cv_split_indices, n_folds_total, normalized=False)
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			
			# print results and change relative to full model
			print(mouseID)
			print(category)
			print('no weights r2: ', np.nanmean(np.nanmedian(r2, axis=1)))
			print('change from full: ', np.nanmean(np.nanmedian(r2-full_model_r2, axis=1)))

			# save r2 values
			if normalized:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_no_weights_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy', r2)
			else:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_no_weights_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy', r2)
			print()

if permute_edges:
	# set cv details
	split_method=='trial'
	splitter = 'KFold' 
	cv_folds=2
	n_repeats=10

	n_permutations=100
	
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		fig_save_dir =  mouse_dir + 'carry_analysis/GLM/figures/'
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
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				temp_spks[carry_i] = reg_spks[:, (carry-t_pre-lag):(carry+t_post-lag)]
				category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
				temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
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
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)

		# index edges by if the Pearson Correlation passed the null
		sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl')
		active_edge_bool = np.zeros(len(np.zeros((n_neurons, n_neurons))[np.triu_indices(n_neurons, 1)]), dtype=bool)
		active_edge_bool[sig_weight_indices['active']]=True
		empty_edge_bool = np.zeros(len(active_edge_bool), dtype=bool)
		empty_edge_bool[sig_weight_indices['empty']]=True

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		for category in ['active', 'empty']:
			if category=='active':
				label=0
				weight_bool=active_edge_bool
			else:
				label=1
				weight_bool=empty_edge_bool

			# convert weight bool into boolean matrix of significant FCs
			graph_like_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_edge_bool[np.triu_indices(n_neurons, 1)] = weight_bool
			graph_like_edge_bool += graph_like_edge_bool.T

			# grab category-specific data
			category_idx = behavior[where_not_NaN, -1]==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			
			# import cv split indices
			if splitter=='Shuffle':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
				n_folds_total = cv_folds
			elif splitter=='KFold':
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
				n_folds_total = cv_folds*n_repeats

			# run models
			if normalized:
				full_permute_save_path = save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy'
				weights_only_save_path = save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy'
				all_edges_r2, sig_edges_r2 = parallelized_weight_permutation(cat_features, cat_neural_data, cv_split_indices, n_folds_total, n_permutations, graph_like_edge_bool, weights_edges_save_path = full_permute_save_path, weights_save_path=weights_only_save_path, normalized=True)
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy')
			else:
				full_permute_save_path = save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy'
				weights_only_save_path = save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy'
				all_edges_r2, sig_edges_r2 = parallelized_weight_permutation(cat_features, cat_neural_data, cv_split_indices, n_folds_total, n_permutations, graph_like_edge_bool, weights_edges_save_path = full_permute_save_path, weights_save_path=weights_only_save_path, normalized=False)
				full_model_r2 = np.load(save_dir + f'{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat.npy')
			
			# print results and change relative to full
			print(mouseID)
			print(category)
			print('full permuted weights and edges r2: ', np.nanmean(np.nanmedian(np.nanmean(all_edges_r2, axis=2), axis=1)))
			print('change from full: ', np.nanmean(np.nanmedian(np.nanmean(all_edges_r2, axis=2)-full_model_r2, axis=1)))
			print('permuted weights r2: ', np.nanmean(np.nanmedian(np.nanmean(sig_edges_r2, axis=2), axis=1)))
			print('change from full: ', np.nanmean(np.nanmedian(np.nanmean(sig_edges_r2, axis=2)-full_model_r2, axis=1)))

			# save r2 values
			if normalized:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy', all_edges_r2)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy', sig_edges_r2)
			else:
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy', all_edges_r2)
				np.save(save_dir + f'{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy', sig_edges_r2)
			print()

if apply_to_drops:
	# has to be lag 0 model !!
	# collect the spikes for drops as pre- and post-drop separately
	split_method=='timepoint' # doesn't have to be trial since testing on entirely held out set of trials (drops)
	splitter = 'Shuffle' 
	cv_folds=10
	test_size=0.3
	for mouseID in mice:
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		save_dir = mouse_dir + 'carry_analysis/GLM/'
		
		if not os.path.isdir(save_dir + 'figures/drops/'):
			os.mkdir(save_dir + 'figures/drops/')
			os.mkdir(save_dir + 'figures/drops/active/')
			os.mkdir(save_dir + 'figures/drops/empty/')

		reg_inds = src.utils.load_registered_cells(mouse_dir, days)

		neural_activity_by_trial = [[] for day in days]
		predrop_neural_activity_by_trial = [[] for day in days]
		postdrop_neural_activity_by_trial = [[] for day in days]
		behavior = [[] for day in days]
		predrop_behavior = [[] for day in days]
		postdrop_behavior = [[] for day in days]
		day_labels = [[] for day in days]
		trial_labels = [[] for day in days]
		trial_like_day_labels = [[] for day in days]
		trial_like_cat_labels = [[] for day in days]
		trial_counter=0
		for i, day in enumerate(days):
			print(day)
			# load in the Cascade spikes
			calcium_data_path = mouse_dir + day
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
			
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
			try:	
				drop_times = np.load(s2p_fld + 'calcium_drop_times.npy')
			except:
				drop_times = []

			# load in the kinematic data
			paw_centroid = np.load(s2p_fld+'paw_centroid_interpolated.npy')
			aperature = np.load(s2p_fld+'aperature_interpolated_conservative.npy')
			splay = np.load(s2p_fld+'d2d3_splay_interpolated_conservative.npy')
			orientation = np.load(s2p_fld+'relative_y_normal_to_palm_interpolated.npy')
			# and combine
			kinematic_data = np.vstack((paw_centroid, aperature, splay, orientation)) # now shape=(K, time) where K=6 kinematic variables

			# get trial-specific neural data and behavioral data
			# this is importantly different than other points in this script because there is no lag!!
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

			if len(drop_times)>0:
				# get the same but for pre-drop timepoints
				# this is importantly different than other points in this script because there is no lag!!
				temp_spks = [[] for time in drop_times]
				temp_behavior = [[] for time in drop_times]
				for carry_i, carry in enumerate(drop_times.astype(int)):
					temp_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry)]
					temp_behavior[carry_i] = kinematic_data[:, (carry-t_pre):(carry)]
				predrop_neural_activity_by_trial[i] = np.hstack(temp_spks)
				predrop_behavior[i] = np.hstack(temp_behavior)

				# get the same but for post-drop timepoints
				# this is importantly different than other points in this script because there is no lag!!
				temp_spks = [[] for time in drop_times]
				temp_behavior = [[] for time in drop_times]
				for carry_i, carry in enumerate(drop_times.astype(int)):
					temp_spks[carry_i] = reg_spks[:, (carry):(carry+t_post)]
					temp_behavior[carry_i] = kinematic_data[:, (carry):(carry+t_post)]
				postdrop_neural_activity_by_trial[i] = np.hstack(temp_spks)
				postdrop_behavior[i] = np.hstack(temp_behavior)

			else:
				print(len(drop_times)) # verify that it is actually 0

		# combine data across days
		neural_activity = np.hstack(neural_activity_by_trial)
		behavior = np.hstack(behavior).T
		predrop_neural_activity = np.hstack([n for n in predrop_neural_activity_by_trial if len(n)>0])
		predrop_behavior = np.hstack([n for n in predrop_behavior if len(n)>0]).T
		postdrop_neural_activity = np.hstack([n for n in postdrop_neural_activity_by_trial if len(n)>0])
		postdrop_behavior = np.hstack([n for n in postdrop_behavior if len(n)>0]).T
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
		n_neurons = activity_nonan.shape[0]
		print('n cells: ', n_neurons)
		predrop_activity_nonan = predrop_neural_activity[~np.isnan(neural_activity).any(axis=1), :]
		postdrop_activity_nonan = postdrop_neural_activity[~np.isnan(neural_activity).any(axis=1), :]

		# index edges by if the Pearson Correlation passed the null
		sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl')
		active_edge_bool = np.zeros(len(np.zeros((n_neurons, n_neurons))[np.triu_indices(n_neurons, 1)]), dtype=bool)
		active_edge_bool[sig_weight_indices['active']]=True
		empty_edge_bool = np.zeros(len(active_edge_bool), dtype=bool)
		empty_edge_bool[sig_weight_indices['empty']]=True

		# import weights for active and empty
		active_graph = np.load(s2p_fld + f'active_FN_pearson_corr.npy')
		empty_graph = np.load(s2p_fld + f'empty_FN_pearson_corr.npy')

		# remove timepoints with NaNs
		where_not_NaN = ~np.isnan(behavior).any(axis=1)
		features = behavior[where_not_NaN, :-1]
		neuronal_data = activity_nonan[:, where_not_NaN]
		day_labels = day_labels[where_not_NaN]
		trial_labels = trial_labels[where_not_NaN]
		if not len(np.unique(trial_labels))==len(trial_like_day_labels):
			warnings.warn('removing NaNs removed at least one entire trial')
			og_trial_count = len(trial_like_day_labels)
			trial_like_day_labels = trial_like_day_labels[np.unique(trial_labels).astype(int)]
			trial_like_cat_labels = trial_like_cat_labels[np.unique(trial_labels).astype(int)]
			print(f'reduced to {len(trial_like_day_labels)} of {og_trial_count} original trials')
		print(features.shape)

		predrop_where_not_NaN = ~np.isnan(predrop_behavior).any(axis=1)
		predrop_features = predrop_behavior[predrop_where_not_NaN, :]
		predrop_neural_data = predrop_activity_nonan[:, predrop_where_not_NaN]
		print(np.sum(~np.isnan(predrop_behavior).any(axis=1)), '/', len(predrop_where_not_NaN))
		postdrop_where_not_NaN = ~np.isnan(postdrop_behavior).any(axis=1)
		postdrop_features = postdrop_behavior[postdrop_where_not_NaN, :]
		postdrop_neural_data = postdrop_activity_nonan[:, postdrop_where_not_NaN]
		print(np.sum(~np.isnan(postdrop_behavior).any(axis=1)), '/', len(postdrop_where_not_NaN))


		for category in ['active', 'empty']:
			if category=='active':
				label=0
				weight_bool=active_edge_bool
			else:
				label=1
				weight_bool=empty_edge_bool

			# convert boolean array into matrix
			graph_like_edge_bool = np.zeros((n_neurons, n_neurons), dtype=bool)
			graph_like_edge_bool[np.triu_indices(n_neurons, 1)] = weight_bool
			graph_like_edge_bool += graph_like_edge_bool.T

			# grab category-specific data
			category_idx = behavior[where_not_NaN, -1]==label
			trial_like_cat_idx = trial_like_cat_labels==label
			cat_features = features[category_idx, :]
			cat_day_labels = day_labels[category_idx]
			cat_neural_data = neuronal_data[:, category_idx]
			cat_trial_labels = trial_labels[category_idx]
			
			if split_method=='trial':
				if splitter=='Shuffle':
					# import cv split indices
					cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{test_size}cv_splits_{cv_folds}_folds_distr_strat.pkl')
					# run models
					if normalized:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds, fig_save_dir=save_dir + f'figures/drops/{category}/', normalized=True)
					else:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds, fig_save_dir=save_dir + f'figures/drops/{category}/', normalized=False)
				elif splitter=='KFold':
					# import cv split indices
					cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_trial_{cv_folds}Fold_cv_splits_{n_repeats}_repeats_distr_strat.pkl')
					# run models
					if normalized:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds*n_repeats, fig_save_dir=None, normalized=True)
					else:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds*n_repeats, fig_save_dir=None, normalized=False)
			elif split_method=='timepoint':
				# import cv split indices
				cv_split_indices = src.IO.load_pickle(save_dir + f'{category}_timepoint_{splitter}_cv_splits_{cv_folds}_folds.pkl')
				# run models
				if splitter=='Shuffle':
					if normalized:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds, fig_save_dir=None, normalized=True)
					else:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds, fig_save_dir=None, normalized=False)
				elif splitter=='KFold':
					if normalized:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds*n_repeats, fig_save_dir=None, normalized=True)
					else:
						r2_values, predrop_r2, postdrop_r2 = drop_applied_ridge_regression(cat_features, cat_neural_data, graph_like_edge_bool, predrop_neural_data, predrop_features, postdrop_neural_data, postdrop_features, cv_split_indices, cv_folds*n_repeats, fig_save_dir=None, normalized=False)

			# print results
			print(mouseID)
			print(category)

			print('median full model r2: ', np.median(np.mean(r2_values, axis=1))) # note that this is generally higher than the full model because of train/test bleed
			print('median pre-drop r2: ', np.median(np.mean(predrop_r2, axis=1)))
			print('median post-drop r2: ', np.median(np.mean(postdrop_r2, axis=1)))
			print()

			# save r2 values
			if split_method=='timepoint':
				if normalized:
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_predrop_r2_values_NORMALIZED.npy', predrop_r2)
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_postdrop_r2_values_NORMALIZED.npy', postdrop_r2)
				else:
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_predrop_r2_values.npy', predrop_r2)
					np.save(save_dir + f'{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_postdrop_r2_values.npy', postdrop_r2)
			elif split_method=='trial':
				if normalized:
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_predrop_r2_values_NORMALIZED.npy', predrop_r2)
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_postdrop_r2_values_NORMALIZED.npy', postdrop_r2)
				else:
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_r2_values.npy', r2_values)
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_predrop_r2_values.npy', predrop_r2)
					np.save(save_dir + f'{category}_kinematics_and_coupling_trial_{splitter}_split_ridge_regression_postdrop_r2_values.npy', postdrop_r2)
