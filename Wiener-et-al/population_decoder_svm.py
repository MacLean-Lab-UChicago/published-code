'''
decode via a linear SVM from the full population (rather than cell by cell)
and augment by decodable cells or non-decodable cells, then compare across all of these
'''
import numpy as np 
import src.utils
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
data_dir = Path.cwd() / 'data'

warnings.filterwarnings("ignore", module="sklearn")

def subsample_dataset_into_sets(x, labels, verbose=False):
    class_0 = x[labels==0, :]
    class_1 = x[labels==1, :]
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
                if leftover!=0:
	                assert leftover == len(class_0_idxs), 'leftover not equal to smaller class'
	                class_0_idxs.extend(np.random.choice(np.sum(labels==0), size = n_less-leftover, replace=False))
            elif larger_class==1:
                class_1_idxs = [i for i in range(np.sum(labels==1)) if i not in larger_class_indices]
                if leftover!=0:
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
        
        subsampled_feat_0 = class_0[class_0_idxs, :]
        subsampled_feat_1 = class_1[class_1_idxs, :]
        subsampled_x_temp = np.append(subsampled_feat_0, subsampled_feat_1, axis=0)
        subsampled_labels_temp = np.append(np.zeros(n_less), np.ones(n_less))
        assert subsampled_x_temp.shape[0]==len(subsampled_labels_temp), 'features and labels different n_trials'
        subsampled_x.append(subsampled_x_temp)
        subsampled_labels.append(subsampled_labels_temp)
    return subsampled_x, subsampled_labels

def indices_random_subsample(array_size, n_less, n_resamples=100):
	indices = [[] for i in range(n_resamples)]
	for i in range(n_resamples):
		indices[i] = np.random.randint(array_size, size=n_less)
	return(indices)


# mouseID='mouse39'
mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
avg_fr_options = [False, True]
cell_pop = ['non-decodable', 'decodable', 'full']
downsample = True
run_dimensionality_controls=True
n_folds=10
test_size=0.1

t_pre = 4 #frames (30 fps) 
t_post = 4 #frames
time_in_trial = t_pre+t_post

save_path = data_dir + '/results/cross_mouse_results/'

# initialize nested lists for saving all mouse decoding at once
accuracy = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]
error = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]
CI_95 = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]

if run_dimensionality_controls:
	ctrl_accuracy = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]
	ctrl_error = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]
	ctrl_CI_95 = [np.zeros(len(avg_fr_options)*len(cell_pop)) for mouse in mice]

for mouse_i, mouseID in enumerate(mice):
	drive = data_dir + '/neural'
	mouse_dir = drive + '/' + mouseID + '/'
	days = src.IO.get_carry_days(mouseID)

	reg_inds = src.utils.load_registered_cells(mouse_dir, days)

	which_model_count = 0
	for avg_fr in avg_fr_options:
		print(avg_fr)
		for population in cell_pop:
			print(population)
			active_spks = [[] for i in range(len(days))]
			empty_spks = [[] for i in range(len(days))]

			for i, day in enumerate(days):
				# load in the Cascade spikes
				calcium_data_path = mouse_dir + day
				s2p_fld = calcium_data_path + '/'
				
				spks = np.load(s2p_fld + 'cascade_spks.npy')
				# index by the registered cells
				reg_spks = spks[reg_inds[i], :]
				n_neurons=reg_spks.shape[0]

				# load in the behavioral times
				carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
				carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
				active_carry_times = carry_times[carry_labels==1]
				empty_carry_times = carry_times[carry_labels==2]

				if avg_fr:
					# grab average firing rate of each neuron for each trial
					temp_a_spks = np.zeros((n_neurons, len(a_carry_times)))
					temp_e_spks = np.zeros((n_neurons, len(empty_carry_times)))
					for carry_i, carry in enumerate(a_carry_times.astype(int)):
						temp_a_spks[:, carry_i] = np.sum(reg_spks[:, (carry-t_pre):(carry+t_post)], axis=1)/time_in_trial 
					for carry_i, carry in enumerate(empty_carry_times.astype(int)):
						temp_e_spks[:, carry_i] = np.sum(reg_spks[:, (carry-t_pre):(carry+t_post)], axis=1)/time_in_trial
					active_spks[i]=temp_a_spks
					empty_spks[i]=temp_e_spks
				else:
					# grab instantaneous firing rate of each neuron for each trial across all time points
					temp_s_spks = np.zeros((n_neurons, len(active_carry_times), time_in_trial))
					temp_d_spks = np.zeros((n_neurons, len(empty_carry_times), time_in_trial))
					for carry_i, carry in enumerate(active_carry_times.astype(int)):
						temp_s_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					for carry_i, carry in enumerate(empty_carry_times.astype(int)):
						temp_d_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					active_spks[i]=temp_s_spks
					empty_spks[i]=temp_d_spks

			# make NeuronsxTrialsxTime array of neural activity (or if avg fr then NeuronsxTrials)
			active = np.concatenate(active_spks, axis=1)
			empty = np.concatenate(empty_spks, axis=1)
			print(active.shape)

			n_active = active.shape[1]
			n_empty = empty.shape[1]

			decodable_neurons = np.load(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy')
			n_decodable_neurons = len(decodable_neurons)
			non_decode_idx = np.ones(n_neurons, dtype=bool)
			non_decode_idx[decodable_neurons]=False
			n_nondecode_neurons = len(non_decode_idx)

			n_less = min([n_decodable_neurons, n_nondecode_neurons])
			if run_dimensionality_controls:
				print(n_less)

			if population=='decodable':
				if avg_fr:
					active = active[decodable_neurons, :]
					empty = empty[decodable_neurons, :]
				else:
					active = active[decodable_neurons, :, :]
					empty = empty[decodable_neurons, :, :]
				n_neurons = len(decodable_neurons)
				print(n_neurons)
				print(active.shape)
			elif population=='non-decodable':
				if avg_fr:
					active = active[non_decode_idx, :]
					empty = empty[non_decode_idx, :]
				else:
					active = active[non_decode_idx, :, :]
					empty = empty[non_decode_idx, :, :]
				n_neurons = np.sum(non_decode_idx)
				print(n_neurons)
				print(active.shape)

			# rearrange arrays such that trials are the 0th axis in a 2d array
			if avg_fr:
				active = active.T
				empty = empty.T
			else:
				active = np.transpose(active, axes=[1, 0, 2])
				active = active.reshape((n_active, n_neurons*time_in_trial))
				empty = np.transpose(empty, axes=[1, 0, 2])
				empty = empty.reshape((n_empty, n_neurons*time_in_trial))

			features = np.concatenate((active, empty), axis=0) # combine along trials axis
			labels = np.append(np.ones(n_active), np.zeros(n_empty)) # binary trial labels
			assert features.shape[0]==len(labels), f'{features.shape[0]} {len(labels)}'

			# remove any cells with NaNs
			n_NaNs = np.sum(np.isnan(features).any(axis=0))
			feature_data = features[:, ~np.isnan(features).any(axis=0)]
			print(features.shape)
			print(feature_data.shape)
				
			SVM = svm.LinearSVC(random_state=25)
			test_accuracy = []

			if downsample:
				feature_sets, label_sets = subsample_dataset_into_sets(feature_data, labels)
			else:
				feature_sets = [feature_data]
				label_sets = [labels]

			for feat_data, lab in zip(feature_sets, label_sets):
				cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size)
				for fold, (train_idx, test_idx) in enumerate(cv.split(feat_data, lab)):
					features_train = feat_data[train_idx, :]
					features_test = feat_data[test_idx, :]
					labels_train = lab[train_idx]
					labels_test = lab[test_idx]

					distributions = dict(C=np.logspace(0, 5, num=10), penalty=['l1', 'l2'])
					clf = GridSearchCV(SVM, distributions, scoring='balanced_accuracy')
					search = clf.fit(features_train, labels_train)
					# print(search.best_params_)
					# print(search.best_score_)
					model = svm.LinearSVC(C=search.best_params_['C'], penalty=search.best_params_['penalty'], random_state=25)
					model.fit(features_train, labels_train)
					predicted_labels = model.predict(features_test)
					test_accuracy.append(metrics.accuracy_score(labels_test, predicted_labels))

			if downsample:
				chance=0.5
			else:
				chance = np.max([n_active, n_empty])/(n_active+n_empty)
			mean_accuracy = np.mean(test_accuracy)
			accuracy_sem = stats.sem(test_accuracy)
			CI = 1.96*accuracy_sem
			
			accuracy[mouse_i][which_model_count] = mean_accuracy
			error[mouse_i][which_model_count] = accuracy_sem
			CI_95[mouse_i][which_model_count] = CI

			print(mouseID)
			print(chance)
			print(n_active)
			print(n_empty)
			print(mean_accuracy, '+-', CI)

			if downsample:
				print('downsampled')
			elif oversample:
				print('oversampled')
			else:
				print('not balanced')

			# save the full test accuracies array for t-testing later
			if avg_fr:
				np.save(s2p_fld + f'avg_fr_{population}_svm_accuracies_test_{test_size}.npy', test_accuracy)
			else:
				np.save(s2p_fld + f'instant_fr_{population}_svm_accuracies_test_{test_size}.npy', test_accuracy)

			if run_dimensionality_controls:
				if avg_fr:
					subsample_to = n_less-n_NaNs
				else:
					subsample_to = n_less*time_in_trial-n_NaNs

				neuron_subsample_indices = indices_random_subsample(feature_data.shape[1], subsample_to)
				test_accuracy = []
				SVM = svm.LinearSVC(random_state=25)
				for control_subsample in neuron_subsample_indices:
					control_feature_data = feature_data[:, control_subsample]
			
					if downsample:
						feature_sets, label_sets = subsample_dataset_into_sets(control_feature_data, labels)
					elif oversample:
						feature_sets, label_sets = oversample_into_sets(control_feature_data, labels, n_resamples=5)
					else:
						feature_sets = [control_feature_data]
						label_sets = [labels]

					for feat_data, lab in zip(feature_sets, label_sets):
						cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size)
						for fold, (train_idx, test_idx) in enumerate(cv.split(feat_data, lab)):
							features_train = feat_data[train_idx, :]
							features_test = feat_data[test_idx, :]
							labels_train = lab[train_idx]
							labels_test = lab[test_idx]

							distributions = dict(C=np.logspace(0, 5, num=10), penalty=['l1', 'l2'])
							clf = GridSearchCV(SVM, distributions, scoring='balanced_accuracy')
							search = clf.fit(features_train, labels_train)
							# print(search.best_params_)
							# print(search.best_score_)
							model = svm.LinearSVC(C=search.best_params_['C'], penalty=search.best_params_['penalty'], random_state=25)
							model.fit(features_train, labels_train)
							predicted_labels = model.predict(features_test)
							test_accuracy.append(metrics.accuracy_score(labels_test, predicted_labels))

					if downsample:
						chance=0.5
					elif oversample:
						chance=0.5
					else:
						chance = np.max([n_active, n_empty])/(n_active+n_empty)
					mean_accuracy = np.mean(test_accuracy)
					accuracy_sem = stats.sem(test_accuracy)
					CI = 1.96*accuracy_sem
					
					ctrl_accuracy[mouse_i][which_model_count] = mean_accuracy
					ctrl_error[mouse_i][which_model_count] = accuracy_sem
					ctrl_CI_95[mouse_i][which_model_count] = CI

				if avg_fr:
					np.save(s2p_fld + f'avg_fr_{population}_CONTROL_DOWNSAMPLE_svm_accuracies_test_{test_size}.npy', test_accuracy)
				else:
					np.save(s2p_fld + f'instant_fr_{population}_CONTROL_DOWNSAMPLE_svm_accuracies_test_{test_size}.npy', test_accuracy)

			which_model_count+=1


np.save(save_path + f'SVM_accuracies_test_{test_size}.npy', accuracy)
np.save(save_path + f'SVM_error_test_{test_size}', error)
np.save(save_path + f'95_CI_test_{test_size}', CI_95)

if run_dimensionality_controls:
	np.save(save_path + f'Control_SVM_accuracies_test_{test_size}.npy', ctrl_accuracy)
	np.save(save_path + f'Control_SVM_error_test_{test_size}.npy', ctrl_error)
	np.save(save_path + f'Control_95_CI_test_{test_size}.npy', ctrl_CI_95)