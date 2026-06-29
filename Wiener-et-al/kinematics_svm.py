'''
Linear kernel SVM Classifier on the kinemtic data to classify carry trial type

requires (per mouse):
	paw_centroid_interpolated, aperature_interpolated, d2d3_splay_interpolated, relative_y_normal_to_palm_interpolated
	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
outputs:
	{mouseID}_kinematics_svm_accuracies_test_.1.npy, {mouseID}_gross_kinematics_svm_accuracies_test_.1.npy, {mouseID}_fine_kinematics_svm_accuracies_test_.1.npy 
		(numpy array of all test accuracies for each mouse for full, gross, and fine kinematics svm decoders at test size .1)
	gross_SVM_accuracies, gross_SVM_error, gross_95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for gross kinematics svm decoders)
	fine_SVM_accuracies, fine_SVM_error, fine_95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for fine kinematics svm decoders)
	SVM_accuracies, SVM_error, 95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for full kinematics svm decoders)
	mouse-averaged kinematic svm accuracy.pdf (bar plot of kinematic SVM accuracies, separated by gross, fine, and full)
'''
import numpy as np
import functions
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
import src.IO
warnings.filterwarnings("ignore", module="sklearn")

def subsample_dataset_into_sets(x, labels, verbose=False):
	class_0 = x[labels==0, :]
	class_1 = x[labels==1, :]
	sizes = [np.sum(labels==0), np.sum(labels==1)]
	n_less = np.min(sizes)
	n_more = max(sizes)
	n_subsamples = np.ceil(n_more/n_less).astype(int)
	leftover = n_more%n_less
	class_0_indices = []
	class_1_indices = []
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
		
		class_0_indices.append(class_0_idxs)
		class_1_indices.append(class_1_idxs)
	return class_0_indices, class_1_indices

def get_features_from_subsampling_indices(class_0_idxs, class_1_idxs, x, labels):
	class_0 = x[labels==0, :]
	class_1 = x[labels==1, :]
	#print(class_0.shape)

	sizes = [np.sum(labels==0), np.sum(labels==1)]
	n_less = np.min(sizes)

	subsampled_feat_0 = class_0[class_0_idxs, :]
	subsampled_feat_1 = class_1[class_1_idxs, :]
	#print(subsampled_feat_0.shape, subsampled_feat_1.shape)
	subsampled_x = np.append(subsampled_feat_0, subsampled_feat_1, axis=0)
	subsampled_labels = np.append(np.zeros(n_less), np.ones(n_less))
	assert subsampled_x.shape[0]==len(subsampled_labels), f'{subsampled_x.shape[0]} features and {len(subsampled_labels)} labels different n_trials'
	return subsampled_x, subsampled_labels

def run_SVM(features_train, features_test, labels_train, labels_test):
	distributions = dict(C=np.logspace(0, 5, num=10), penalty=['l1', 'l2'])
	clf = GridSearchCV(SVM, distributions, scoring='balanced_accuracy')
	search = clf.fit(features_train, labels_train)
	# print(search.best_params_)
	# print(search.best_score_)
	model = svm.LinearSVC(C=search.best_params_['C'], penalty=search.best_params_['penalty'], random_state=25)
	model.fit(features_train, labels_train)
	predicted_labels = model.predict(features_test)
	accuracy = metrics.accuracy_score(labels_test, predicted_labels)
	return accuracy

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
downsample = True
n_folds = 10 # cross-validation folds
test_size=0.1

t_pre=4
t_post=4
time_in_trial=t_pre+t_post

save_path = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/kinematics_3d/active_grasp/SVM_decoder/'
calculate=True
plot=True

if calculate:
	# initialize arrays to save out accuracies, error, and confidence intervals for all mice
	accuracy = np.zeros(len(mice))
	error = np.zeros(len(mice))
	CI_95 = np.zeros(len(mice))

	gross_accuracy = np.zeros(len(mice))
	gross_error = np.zeros(len(mice))
	gross_CI_95 = np.zeros(len(mice))

	fine_accuracy = np.zeros(len(mice))
	fine_error = np.zeros(len(mice))
	fine_CI_95 = np.zeros(len(mice))

	# iterate over mice
	for mouse_i, mouseID in enumerate(mice):
		days = src.IO.get_carry_days(mouseID)
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'

		behavior_by_trial = [[] for day in days]
		gross_by_trial = [[] for day in days]
		fine_by_trial = [[] for day in days]
		trial_day_labels = []
		category_labels = []
		for i, day in enumerate(days):
			calcium_data_path = mouse_dir + day
			if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
				s2p_fld = calcium_data_path + '/'
			else:
				s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'

			# load in the behavioral times
			carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
			carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
			carry_times = carry_times[carry_labels!=0]
			carry_labels = carry_labels[carry_labels!=0]
			category_labels.extend(carry_labels)
			category_labels_by_time = np.zeros((len(carry_times)*time_in_trial))

			# load in the kinematic data
			paw_centroid = np.load(s2p_fld+'paw_centroid_interpolated.npy')
			aperature = np.load(s2p_fld+'aperature_interpolated.npy')
			splay = np.load(s2p_fld+'d2d3_splay_interpolated.npy')
			orientation = np.load(s2p_fld+'relative_y_normal_to_palm_interpolated.npy')
			# and combine
			kinematic_data = np.vstack((paw_centroid, aperature, splay, orientation)) # now shape=(K, time) where K=6 kinematic variables

			# get trial-specific neural data and behavioral data
			trial_day_behavior = np.zeros((kinematic_data.shape[0], len(carry_times), time_in_trial))
			for carry_i, carry in enumerate(carry_times.astype(int)):
				trial_day_behavior[:, carry_i, :] = kinematic_data[:, (carry-t_pre):(carry+t_post)]
			behavior_by_trial[i] = trial_day_behavior
			trial_day_labels.extend(np.ones(len(carry_times))*i)

		behavior_by_trial = np.concatenate(behavior_by_trial, axis=1)

		trial_day_labels = np.array(trial_day_labels)
		category_labels = np.array(category_labels)

		# remove any trials with kinematic NaNs
		trials_with_NaNs = np.isnan(behavior_by_trial).any(axis=(0, 2))
		print(f'{np.sum(trials_with_NaNs)}/{len(trials_with_NaNs)} trials have NaNs, removed')

		# re-shape the predictors into the necessary dimensions for modelling
		trial_labels = category_labels[~trials_with_NaNs]-1
		kinematic_predictors = np.transpose(behavior_by_trial[:, ~trials_with_NaNs, :], axes=[1, 0, 2])
		kinematic_predictors = kinematic_predictors.reshape((len(trial_labels), -1))
		gross_kinematic_predictors = np.transpose(behavior_by_trial[:3, ~trials_with_NaNs, :], axes=[1, 0, 2])
		gross_kinematic_predictors = gross_kinematic_predictors.reshape((len(trial_labels), -1))
		fine_kinematic_predictors = np.transpose(behavior_by_trial[3:, ~trials_with_NaNs, :], axes=[1, 0, 2])
		fine_kinematic_predictors = fine_kinematic_predictors.reshape((len(trial_labels), -1))
		assert kinematic_predictors.shape[1]==48, f'{kinematic_predictors.shape[1]} features, reshaping gone wrong'
		assert gross_kinematic_predictors.shape[1]==24, f'{gross_kinematic_predictors.shape[1]} features, reshaping gone wrong'
		assert fine_kinematic_predictors.shape[1]==24, f'{fine_kinematic_predictors.shape[1]} features, reshaping gone wrong'
		print(np.sum(trial_labels==0), ' active grasp trials')
		print(np.sum(trial_labels==1), ' dry carry trials')

		# initialize the SVM
		SVM = svm.LinearSVC(random_state=25)
		test_accuracy = []
		gross_test_accuracy = []
		fine_test_accuracy = []

		# make downsampling splits
		if downsample:
			class_0_indices, class_1_indices = subsample_dataset_into_sets(kinematic_predictors, trial_labels)
		else:
			class_0_indices, class_1_indices = [np.arange(np.sum(trial_labels==0))], [np.arange(np.sum(trial_labels==1))]
		for class_0, class_1 in zip(class_0_indices, class_1_indices):
			# get feature and label splits for each downsample
			feat_data, lab = get_features_from_subsampling_indices(class_0, class_1, kinematic_predictors, trial_labels)
			gross_feat_data, lab = get_features_from_subsampling_indices(class_0, class_1, gross_kinematic_predictors, trial_labels)
			fine_feat_data, lab = get_features_from_subsampling_indices(class_0, class_1, fine_kinematic_predictors, trial_labels)
			
			# shuffle split trials
			cv = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size)
			for fold, (train_idx, test_idx) in enumerate(cv.split(feat_data, lab)):
				# grab training and testing data
				features_train = feat_data[train_idx, :]
				features_test = feat_data[test_idx, :]
				labels_train = lab[train_idx]
				labels_test = lab[test_idx]

				# run the decoder
				a = run_SVM(features_train, features_test, labels_train, labels_test)
				test_accuracy.append(a)

				# repeat for gross kinematics
				gross_features_train = gross_feat_data[train_idx, :]
				gross_features_test = gross_feat_data[test_idx, :]

				a = run_SVM(gross_features_train, gross_features_test, labels_train, labels_test)
				gross_test_accuracy.append(a)

				# repeat for fine kinematics
				fine_features_train = fine_feat_data[train_idx, :]
				fine_features_test = fine_feat_data[test_idx, :]

				a = run_SVM(fine_features_train, fine_features_test, labels_train, labels_test)
				fine_test_accuracy.append(a)

		chance=0.5
		mean_accuracy = np.mean(test_accuracy)
		accuracy_sem = stats.sem(test_accuracy)
		CI = 1.96*accuracy_sem
		
		accuracy[mouse_i] = mean_accuracy
		error[mouse_i] = accuracy_sem
		CI_95[mouse_i] = CI

		print(mouseID)
		print(chance)
		print(mean_accuracy, '+-', CI)

		# save out the full test accuracies array for t-testing later
		np.save(save_path + f'{mouseID}_kinematics_svm_accuracies_test_{test_size}.npy', test_accuracy)

		gross_mean_accuracy = np.mean(gross_test_accuracy)
		gross_accuracy_sem = stats.sem(gross_test_accuracy)
		gross_CI = 1.96*gross_accuracy_sem
		
		gross_accuracy[mouse_i] = gross_mean_accuracy
		gross_error[mouse_i] = gross_accuracy_sem
		gross_CI_95[mouse_i] = gross_CI

		print(mouseID)
		print(chance)
		print(gross_mean_accuracy, '+-', gross_CI)

		# save out the full test accuracies array for t-testing later
		np.save(save_path + f'{mouseID}_gross_kinematics_svm_accuracies_test_{test_size}.npy', gross_test_accuracy)

		chance=0.5
		fine_mean_accuracy = np.mean(fine_test_accuracy)
		fine_accuracy_sem = stats.sem(fine_test_accuracy)
		fine_CI = 1.96*fine_accuracy_sem
		
		fine_accuracy[mouse_i] = fine_mean_accuracy
		fine_error[mouse_i] = fine_accuracy_sem
		fine_CI_95[mouse_i] = fine_CI

		print(mouseID)
		print(chance)
		print(fine_mean_accuracy, '+-', fine_CI)

		# save out the full test accuracies array for t-testing later
		np.save(save_path + f'{mouseID}_fine_kinematics_svm_accuracies_test_{test_size}.npy', fine_test_accuracy)

	# save the cross-validated accuracies, errors, and confidence intervals for all mice
	np.save(save_path + 'SVM_accuracies.npy', accuracy)
	np.save(save_path + 'SVM_error.npy', error)
	np.save(save_path + '95_CI.npy', CI_95)

	np.save(save_path + 'gross_SVM_accuracies.npy', gross_accuracy)
	np.save(save_path + 'gross_SVM_error.npy', gross_error)
	np.save(save_path + 'gross_95_CI.npy', gross_CI_95)

	np.save(save_path + 'fine_SVM_accuracies.npy', fine_accuracy)
	np.save(save_path + 'fine_SVM_error.npy', fine_error)
	np.save(save_path + 'fine_95_CI.npy', fine_CI_95)


if plot:
	# plot mouse-averaged accuracies for gross, fine, and combined models
	fine_accuracies = np.load(save_path + 'fine_SVM_accuracies.npy')
	gross_accuracies = np.load(save_path + 'gross_SVM_accuracies.npy')
	accuracies = np.load(save_path + 'SVM_accuracies.npy')

	mean_fine = np.mean(fine_accuracies)
	mean_gross = np.mean(gross_accuracies)
	mean_full = np.mean(accuracies)
	sem_fine = stats.sem(fine_accuracies)
	sem_gross = stats.sem(gross_accuracies)
	sem_full = stats.sem(accuracies)

	fine_vs_gross = stats.ttest_rel(fine_accuracies, gross_accuracies, alternative='greater')
	print(fine_vs_gross)

	by_mouse_a = np.concatenate([gross_accuracies[:, np.newaxis], fine_accuracies[:, np.newaxis], accuracies[:, np.newaxis]], axis=1)

	colors = ['cornflowerblue', 'royalblue', 'mediumblue']
	categories = ['gross kinematics', 'fine kinematics', 'all kinematics']

	fig, ax = plt.subplots()
	ax.bar([0, 2, 4], [mean_gross, mean_fine, mean_full], color=colors, yerr=[sem_gross, sem_fine, sem_full], alpha=0.5, capsize=5)
	ax.set_xticks([0, 2, 4], categories)
	for cat_i, category in enumerate(categories):
		accur = np.sort(by_mouse_a[:, cat_i])
		offset_max=0.1
		if any(np.diff(accur)<.02):
			left=True
			for d, dot in enumerate(accur):
				r = np.random.rand()*offset_max
				if left:
					ax.scatter([0, 2, 4][cat_i]-r, dot, color=colors[cat_i])
					left=False
				else:
					ax.scatter([0, 2, 4][cat_i]+r, dot, color=colors[cat_i])
					left=True
		else:
			ax.scatter(np.ones(len(mice), dtype=object)*[0, 2, 4][cat_i], accur, color=colors[cat_i])

	ax.set_ylim([0.35, 1])
	ax.axhline(0.5, color='k', ls='--', label='chance')
	ax.set_ylabel('accuracy')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	plt.savefig(save_path + 'mouse-averaged kinematic svm accuracy.pdf', dpi=550)
	plt.show()