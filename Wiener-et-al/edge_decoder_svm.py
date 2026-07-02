'''
SVM to decode trial category from instaneous edge weight
'''
import numpy as np
import src.IO
import src.utils
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
import multiprocessing
import functools
from pathlib import Path
data_dir = Path.cwd() / 'data'

warnings.filterwarnings("ignore", module="sklearn")

def subsample_dataset_into_sets(x, labels, verbose=False):
    class_0 = x[:, labels==0, :]
    class_1 = x[:, labels==1, :]
    sizes = [np.sum(labels==0), np.sum(labels==1)]
    n_less = np.min(sizes)
    n_more = max(sizes)
    n_subsamples = np.ceil(n_more/n_less).astype(int)
    leftover = n_more%n_less
    subsampled_x = []
    subsampled_labels = []
    larger_class_indices = []
    if np.sum(labels)>np.sum(labels==0):
        larger_class=1
    else:
        larger_class=0
    for fold in range(n_subsamples):
        #print(larger_class_indices)
        if fold==n_subsamples-1:
            if larger_class==0:
                class_0_idxs = [i for i in range(np.sum(labels==0)) if i not in larger_class_indices]
                assert leftover == len(class_0_idxs), 'leftover not equal to smaller class'
                class_0_idxs.extend(np.random.choice(np.sum(labels==0), size = n_less-leftover, replace=False))
            elif larger_class==1:
                class_1_idxs = [i for i in range(np.sum(labels==1)) if i not in larger_class_indices]
                assert leftover == len(class_1_idxs), 'leftover not equal to smaller class'
                class_1_idxs.extend(np.random.choice(np.sum(labels==1), size = n_less-leftover, replace=False))
        else:
            if larger_class==0:
                possible_idx = [i for i in range(np.sum(labels==0)) if i not in larger_class_indices]
                class_0_idxs = np.random.choice(possible_idx, size = n_less, replace=False)
                larger_class_indices.extend(class_0_idxs)
                class_1_idxs = np.random.choice(np.sum(labels==1), size = n_less, replace=False)
            elif larger_class==1:
                possible_idx = [i for i in range(np.sum(labels==1)) if i not in larger_class_indices]
                class_1_idxs = np.random.choice(possible_idx, size = n_less, replace=False)
                larger_class_indices.extend(class_1_idxs)
                class_0_idxs = np.random.choice(np.sum(labels==0), size = n_less, replace=False)
        
        subsampled_feat_0 = class_0[:, class_0_idxs, :]
        subsampled_feat_1 = class_1[:, class_1_idxs, :]
        subsampled_x_temp = np.append(subsampled_feat_0, subsampled_feat_1, axis=1)
        subsampled_labels_temp = np.append(np.zeros(n_less), np.ones(n_less))
        assert subsampled_x_temp.shape[1]==len(subsampled_labels_temp), 'features and labels different n_trials'
        subsampled_x.append(subsampled_x_temp)
        subsampled_labels.append(subsampled_labels_temp)
    return subsampled_x, subsampled_labels

def svm_worker(edge_idx, edge_correlations, category_labels, time_in_trial, training_idx, testing_idx):
	features = edge_correlations[edge_idx, :, :].reshape((len(category_labels), -1))
	assert features.shape[1] == time_in_trial, 'reshaped wrong'
	features_train = features[training_idx, :]
	labels_train = category_labels[training_idx]
	features_test = features[testing_idx, :]
	labels_test = category_labels[testing_idx]
	SVM = svm.LinearSVC(penalty='l2', random_state=25)
	distributions = dict(C=np.logspace(0, 5, num=10))
	clf = RandomizedSearchCV(SVM, distributions, scoring='balanced_accuracy')
	search = clf.fit(features_train, labels_train)
	# print(search.best_params_)
	# print(search.best_score_)
	model = svm.LinearSVC(C=search.best_params_['C'], penalty='l2', random_state=25)
	model.fit(features_train, labels_train)
	predicted_labels = model.predict(features_test)
	accuracy = metrics.accuracy_score(labels_test, predicted_labels)
	return accuracy

def parallelized_svm(edge_correlations, category_labels, time_in_trial, train_idx, test_idx, n_edges):
	worker_func = functools.partial(
		svm_worker,
		edge_correlations=edge_correlations, 
		category_labels=category_labels, 
		time_in_trial=time_in_trial, 
		training_idx=train_idx, 
		testing_idx=test_idx
	)
	
	# Create a pool and map the work across cores
	with multiprocessing.Pool() as pool:
		# map distributes the n_neurons samples to the worker function
		# results is a list of accuracy values with length n_edges
		results = pool.map(worker_func, range(n_edges))

	return np.array(results)


mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
t_pre=4
t_post=4
time_in_trial = t_pre+t_post
cv_folds=10

for mouseID in mice:
	days = src.IO.get_carry_days(mouseID)
	drive = data_dir + '/neural/'
	mouse_dir = drive + '/' + mouseID + '/'

	# load in cross-day cell registration
	reg_inds = src.utils.load_registered_cells(mouse_dir, days)
	
	# intialize lists to combine data across days
	carry_labels_cross_day = []
	neural_activity_by_trial = [[] for day in days]
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

		# remove drops (carry_labels==0) and reindex to get a boolean for active and empty
		carry_times = carry_times[carry_labels!=0]
		carry_labels = carry_labels[carry_labels!=0]-1
		carry_labels_cross_day.extend(carry_labels)

		# grab the neural data during carries
		temp_spks = np.zeros((n_neurons, len(carry_times), time_in_trial))*np.nan
		for carry_i, carry in enumerate(carry_times.astype(int)):
			temp_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
		neural_activity_by_trial[i] = temp_spks

	# combine data across days
	carry_labels_cross_day = np.array(carry_labels_cross_day)
	neural_activity = np.concatenate(neural_activity_by_trial, axis=1)
	assert neural_activity.shape[1]==len(carry_labels_cross_day), 'concatenated wrong axes'
	neural_activity_by_cell = neural_activity.copy()
	neural_activity_by_cell = neural_activity_by_cell.reshape((n_neurons, -1))
	assert neural_activity_by_cell.shape[1] == len(carry_labels_cross_day)*time_in_trial, 'reshaping gone wrong'

	# mean-subtract and normalize (needed to calculate instant pearson corr)
	mean_neural_activity = np.nanmean(neural_activity_by_cell, axis=1)
	neural_activity_meansubtracted = neural_activity - mean_neural_activity[:, np.newaxis, np.newaxis]
	print(neural_activity_meansubtracted.shape)
	neural_activity_normalized = neural_activity_meansubtracted/np.linalg.norm(neural_activity_by_cell-mean_neural_activity[:, np.newaxis], axis=1)[:, np.newaxis, np.newaxis]
	print(neural_activity_normalized.shape)

	# remove any neurons with NaNs
	activity_nonan = neural_activity_normalized[~np.isnan(neural_activity_normalized).any(axis=(1, 2)), :]
	print(np.sum(np.isnan(neural_activity_normalized).any(axis=(1, 2))), ' NaN-containing cells removed')
	NaN_indices = np.where(np.isnan(neural_activity_normalized).any(axis=(1, 2)))[0]
	red_cells_bool = red_cells[~np.isnan(neural_activity_normalized).any(axis=(1, 2))]
	# print(NaN_indices)
	n_neurons = activity_nonan.shape[0]
	print('n cells: ', n_neurons)

	# calculate instant pearson correlations for all timepoints for all FCs
	instant_edge_correlations = np.zeros((n_neurons, n_neurons, len(carry_labels_cross_day), time_in_trial))*np.nan
	for i in range(n_neurons):
		for j in range(n_neurons):
			if i==j:
				continue
			elif np.isnan(instant_edge_correlations[i, j, 0, 0]):
				instant_edge_correlations[i, j, :, :] = activity_nonan[i, :, :]*activity_nonan[j, :, :]
				instant_edge_correlations[j, i, :, :] = activity_nonan[i, :, :]*activity_nonan[j, :, :]
	edge_correlation_features = instant_edge_correlations[np.triu_indices(n_neurons, 1)[0], np.triu_indices(n_neurons, 1)[1], :, :]
	assert edge_correlation_features.shape[0] == (n_neurons*(n_neurons-1)/2), 'wrong number of edges'
	n_edges = edge_correlation_features.shape[0]

	# downsample data for balanced modeling
	feature_sets, label_sets = subsample_dataset_into_sets(edge_correlation_features, carry_labels_cross_day)
	
	# run models and save accuracy
	accuracies_by_edge = np.zeros((len(label_sets)*cv_folds, n_edges))
	for set_idx, (feat_data, lab) in enumerate(zip(feature_sets, label_sets)):
		print(f'working on set {set_idx+1}/{len(feature_sets)}')
		cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=25)
		for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(lab)), lab)):
			print(f'running fold {fold}')
			accuracies_by_edge[set_idx*cv_folds+fold, :] = parallelized_svm(feat_data, lab, time_in_trial, train_idx, test_idx, n_edges)
	
	# save out accuracy for each mouse
	np.save(s2p_fld + 'svm_decoding_accuracy_by_edge.npy', accuracies_by_edge)

