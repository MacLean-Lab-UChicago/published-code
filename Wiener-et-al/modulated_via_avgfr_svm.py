'''
find modulated cells using an SVM to predict if it is a empty or active grasp
and if it is above chance (including 95% CI), it is modulated
(as in Levy et al 2020, Neuron from Hantman)
'''
import numpy as np 
import src.utils
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
import random

warnings.filterwarnings("ignore", module="sklearn")

def subsample_dataset_into_sets(x, labels, verbose=False):
	'''
	function to subsample the data into balanced sets while preserving all data
	inputs: 
		x: numpy array shape 1xT where T is the number of trials
		labels: binary numpy array of length T
	outputs:
		subsampled_x: nested list of trial feature data
		subsampled_labels: nested list of trial labels
	'''
    class_0 = x[labels==0]
    class_1 = x[labels==1]
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
        
        subsampled_feat_0 = class_0[class_0_idxs]
        subsampled_feat_1 = class_1[class_1_idxs]
        subsampled_x_temp = np.append(subsampled_feat_0, subsampled_feat_1, axis=0)
        subsampled_labels_temp = np.append(np.zeros(n_less), np.ones(n_less))
        assert subsampled_x_temp.shape==subsampled_labels_temp.shape, 'features and labels different n_trials'
        subsampled_x.append(subsampled_x_temp)
        subsampled_labels.append(subsampled_labels_temp)
    return subsampled_x, subsampled_labels

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

downsample = True
n_folds=10

t_pre = 4 #frames (30 fps) 
t_post = 4 #frames
time_in_trial = t_pre+t_post

for mouseID in mice:
	# load in cross-day alignment information
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	days = src.IO.get_carry_days(mouseID)

	reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, days)

	active_spks = [[] for i in range(len(days))]
	empty_spks = [[] for i in range(len(days))]

	for i, day in enumerate(days):
		# load in the Cascade spikes
		calcium_data_path = mouse_dir + day
		if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
			s2p_fld = calcium_data_path + '/'
		else:
			s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
		
		spks = np.load(s2p_fld + 'cascade_spks.npy')
		
		# index by the registered cells
		reg_spks = spks[reg_inds[i], :]
		n_neurons=reg_spks.shape[0]

		# load in the behavioral times
		carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
		carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
		active_carry_times = carry_times[carry_labels==1]
		empty_carry_times = carry_times[carry_labels==2]

		# grab the average neural activity per trial per neuron
		temp_a_spks = np.zeros((n_neurons, len(active_carry_times)))
		temp_e_spks = np.zeros((n_neurons, len(empty_carry_times)))
		for carry_i, carry in enumerate(active_carry_times.astype(int)):
			avg_fr = np.sum(reg_spks[:, (carry-t_pre):(carry+t_post)], axis=1)/time_in_trial
			temp_a_spks[:, carry_i] = avg_fr 
		for carry_i, carry in enumerate(empty_carry_times.astype(int)):
			temp_e_spks[:, carry_i] = np.sum(reg_spks[:, (carry-t_pre):(carry+t_post)], axis=1)/time_in_trial
		active_spks[i]=temp_a_spks
		empty_spks[i]=temp_e_spks

	# make NxT (neurons x trials) array containing neural activity across all days
	active = np.concatenate(active_spks, axis=1)
	empty = np.concatenate(empty_spks, axis=1)

	n_active = active.shape[1]
	n_empty = empty.shape[1]

	features = np.concatenate((active, empty), axis=1) # average firing rates per trial
	labels = np.append(np.ones(active.shape[1]), np.zeros(empty.shape[1])) # binary trial labels

	assert features.shape[1]==len(labels), f'{features.shape[1]} {len(labels)}'

	mod_cells = np.zeros(n_neurons, dtype=bool)
	cell_decoding = np.zeros(n_neurons)*np.nan
	cell_CI = np.zeros(n_neurons)*np.nan
	for cell in range(n_neurons):
		feature_data = features[cell, :]
		assert feature_data.shape[0]==len(labels), f'{feature_data.shape[0]} {len(labels)}'
		if cell>0:
			assert np.logical_not(np.array_equal(feature_data, features[cell-1, :])), 'feature data is identical across cells'
		if np.isnan(feature_data).any():
			print(f'cell {cell} contains NaNs, skipping...')
			continue
		SVM = svm.LinearSVC(penalty='l2', random_state=25)
		test_accuracy = []

		if downsample:
			feature_sets, label_sets = subsample_dataset_into_sets(feature_data, labels)
		else:
			feature_sets = [feature_data]
			label_sets = [labels]

		for feat_data, lab in zip(feature_sets, label_sets):
			cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=.1)
			for fold, (train_idx, test_idx) in enumerate(cv.split(feat_data, lab)):
				features_train = feat_data[train_idx]
				features_test = feat_data[test_idx]
				labels_train = lab[train_idx]
				labels_test = lab[test_idx]
				
				distributions = dict(C=np.logspace(0, 5, num=10))
				clf = GridSearchCV(SVM, distributions, scoring='balanced_accuracy')
				search = clf.fit(features_train.reshape(-1,1), labels_train)

				model = svm.LinearSVC(C=search.best_params_['C'], penalty='l2', random_state=25)
				model.fit(features_train.reshape(-1,1), labels_train)
				predicted_labels = model.predict(features_test.reshape(-1,1))
				test_accuracy.append(metrics.accuracy_score(labels_test, predicted_labels))

		if downsample: 
			chance=0.5
		else:
			chance = np.max([n_active, n_empty])/(n_active+n_empty)
		mean_accuracy = np.mean(test_accuracy)
		cell_decoding[cell] = mean_accuracy
		accuracy_sem = stats.sem(test_accuracy)
		CI = 1.96*accuracy_sem
		cell_CI[cell] = CI
		if (mean_accuracy-CI)>chance:
			mod_cells[cell]=True
			print(mean_accuracy, CI)

	print(mouseID)
	if downsample:
		print('downsampled')
	else:
		print('not balanced')

	print(chance)
	print(n_active)
	print(n_empty)

	print('mod cells: ', np.sum(mod_cells))
	print('proportion: ', np.sum(mod_cells)/n_neurons)
	np.save(s2p_fld + 'SVM_avgfr_active_grasp_modulated_cells.npy', mod_cells)
	np.save(s2p_fld + 'single_cell_SVM_avgfr_accuracies_by_cell.npy', cell_decoding)
	np.save(s2p_fld + 'single_cell_SVM_avgfr_confidence_intervals_by_cell.npy', cell_CI)

