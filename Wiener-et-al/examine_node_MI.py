'''
examine node to trial category mutual information
'''
import numpy as np
import matplotlib.pyplot as plt
import src.IO
import src.utils
from scipy import stats

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
time_decodable_MI = []
trial_decodable_MI = []
decodable_MI = []
non_decodable_MI = []
for mouseID in mice:
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	days = src.IO.get_carry_days(mouseID)
	load_dir = mouse_dir + 'carry_analysis/MI/'
	# reg_inds, red_cells = src.utils.load_registered_and_red_cells(mouse_dir, days)
	s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])
	
	NaN_cells = np.load(s2p_fld + 'NaN_containing_cells_bool.npy')
	decodable_neurons = np.load(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy')
	decodable_neurons_idx = src.utils.adjust_indices_for_NaN_cell_removal(decodable_neurons, np.where(NaN_cells)[0])
	non_decodable_bool = np.ones(len(NaN_cells)-np.sum(NaN_cells), dtype=bool)
	non_decodable_bool[decodable_neurons_idx]=False
	trial_decodable_neurons = np.load(s2p_fld + 'SVM_time_averaged_active_grasp_modulated_cell_indices.npy')
	trial_decode_idx = src.utils.adjust_indices_for_NaN_cell_removal(trial_decodable_neurons, np.where(NaN_cells)[0])
	time_decodable_neurons = np.load(s2p_fld + 'SVM_time_varying_active_grasp_modulated_cell_indices.npy')
	time_decode_idx = src.utils.adjust_indices_for_NaN_cell_removal(time_decodable_neurons, np.where(NaN_cells)[0])

	MI = np.load(load_dir + 'node_trialcat_MI.npy')
	time_decodable_MI.extend(MI[time_decode_idx])
	trial_decodable_MI.extend(MI[trial_decode_idx])
	non_decodable_MI.extend(MI[non_decodable_bool])
	decodable_MI.extend(MI[decodable_neurons_idx])

	fig, ax = plt.subplots()
	ax.ecdf(MI[non_decodable_bool], label='non-decodable neurons')
	ax.ecdf(MI[decodable_neurons_idx], label='decodable neurons')
	ax.set_xlabel('MI')
	ax.set_ylabel('Cumulative Proportion')
	ax.set_xlim([0, np.max(MI)])
	ax.legend(loc='lower right')
	plt.savefig(load_dir + 'cdf of node MI.png')
	plt.close()

	fig, ax = plt.subplots()
	ax.ecdf(MI[non_decodable_bool], label='non-decodable neurons')
	ax.ecdf(MI[time_decode_idx], label='time-varying decodable neurons')
	ax.ecdf(MI[trial_decode_idx], label='time-averaged decodable neurons')
	ax.set_xlabel('MI')
	ax.set_ylabel('Cumulative Proportion')
	ax.set_xlim([0, np.max(MI)])
	ax.legend(loc='lower right')
	plt.savefig(load_dir + 'cdf of node MI by decode cat.png')
	plt.close()

results = stats.mannwhitneyu(non_decodable_MI, decodable_MI, alternative='less')
print(results)

fig, ax = plt.subplots()
ax.ecdf(non_decodable_MI, label='non-decodable neurons')
ax.ecdf(decodable_MI, label='decodable neurons')
ax.set_xlim([0, np.max(np.concatenate([non_decodable_MI, decodable_MI]))])
ax.set_xlabel('MI')
ax.set_ylabel('Cumulative Proportion')
ax.legend(loc='lower right')
plt.show()

fig, ax = plt.subplots()
ax.ecdf(non_decodable_MI, label='non-decodable neurons')
ax.ecdf(time_decodable_MI, label='time-varying decodable neurons')
ax.ecdf(trial_decodable_MI, label='time-averaged decodable neurons')
ax.set_xlim([0, np.max(np.concatenate([non_decodable_MI, decodable_MI]))])
ax.set_xlabel('MI')
ax.set_ylabel('Cumulative Proportion')
ax.legend(loc='lower right')
plt.show()