'''
make pie charts!
'''
import matplotlib.pyplot as plt
import numpy as np
import src.utils
import src.IO

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
plot_cells=False
plot_edges=True
if plot_edges:
	FN_method='pearson_corr'


if plot_cells:
	#keys = ['Increased Firing Rate', 'Decreased Firing Rate', 'Not Informative']
	colors = ['lightcoral', 'firebrick', 'darkred', 'darkgrey']
	load_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
	mod_counts = src.IO.load_pickle(load_dir + 'mod_counts.pkl')
	for m, mouseID in enumerate(mice):
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		fig_dir = mouse_dir + f'carry_analysis/figures/'

		mouse_mod_counts = {key:0 for key in mod_counts.keys()}
		for key in mod_counts.keys():
			mouse_mod_counts[key] = mod_counts[key][m]

		plt.pie(list(mouse_mod_counts.values()), labels=list(mouse_mod_counts.keys()), autopct='%1.1f%%', colors=colors)
		plt.title(f'{mouseID} Cell Proportions')
		plt.savefig(fig_dir + 'mod cells pie chart.png')
		plt.show()

	combined_mod_counts = {key:0 for key in mod_counts.keys()}
	for key in mod_counts.keys():
		combined_mod_counts[key] = np.sum(mod_counts[key])
	print(combined_mod_counts)
	plt.pie(list(combined_mod_counts.values()), labels=list(combined_mod_counts.keys()), autopct='%1.1f%%', colors=colors)
	plt.title(f'Cell Proportions')
	plt.savefig(multimouse_fig_dir + 'mod cells pie chart.pdf', dpi=660)
	plt.show()

	mod_counts = src.IO.load_pickle(load_dir + 'red_mod_counts.pkl')
	for m, mouseID in enumerate(mice):
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		fig_dir = mouse_dir + f'carry_analysis/figures/'

		mouse_mod_counts = {key:0 for key in mod_counts.keys()}
		for key in mod_counts:
			mouse_mod_counts[key] = mod_counts[key][m]

		plt.pie(list(mouse_mod_counts.values()), labels=list(mouse_mod_counts.keys()), autopct='%1.1f%%', colors=colors)
		plt.title(f'{mouseID} Inhibitory Cell Proportions')
		plt.savefig(fig_dir + 'mod I cells pie chart.png')
		plt.show()

	keys = list(mouse_mod_counts.keys())
	combined_mod_counts = {f'PV {key}':0 for key in keys}
	combined_mod_counts.update({f'SST {key}':0 for key in keys})
	for key, label in zip(mod_counts.keys(), keys):
		combined_mod_counts[f'PV {label}'] = np.sum(mod_counts[key][:3])
		combined_mod_counts[f'SST {label}'] = np.sum(mod_counts[key][3:])
	plt.pie(list(combined_mod_counts.values()), labels=list(combined_mod_counts.keys()), autopct='%1.1f%%')
	plt.title(f'Inhibitory Cell Proportions')
	plt.savefig(multimouse_fig_dir + 'mod I cells pie chart.png')
	plt.show()


if plot_edges:
	# pie chart of edges by shared/unique/sign flip/both zero
	shared_unique_counts = {'active unique':0, 'empty unique':0, 'shared':0, 'sign flip':0, 'null':0}
	s_u_keys = ['active unique', 'empty unique', 'shared', 'sign flip', 'null']
	colors = ['royalblue', 'cornflowerblue', 'rebeccapurple', 'mediumorchid', 'lightgrey']
	for mouseID in mice:
		mouse_shared_unique_counts = {'active unique':0, 'empty unique':0, 'shared':0, 'sign flip':0, 'null':0}
		drive = src.IO.get_drive(mouseID)
		mouse_dir = drive + '/' + mouseID + '/'
		fig_dir = mouse_dir + f'carry_analysis/figures/'
		shared_unique = src.IO.load_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_weighted_PDF_10000_{FN_method}.pkl')
		for key_i, key in enumerate(shared_unique.keys()):
			mouse_shared_unique_counts[s_u_keys[key_i]] = len(shared_unique[key])
			shared_unique_counts[s_u_keys[key_i]]+= len(shared_unique[key])
		plt.pie(list(mouse_shared_unique_counts.values()), labels=list(mouse_shared_unique_counts.keys()), autopct='%1.1f%%', colors=colors)
		plt.title(f'{mouseID} Edge Proportions')
		plt.savefig(fig_dir + 'pearson corr edge unique shared pie chart.png')
		plt.close()

	print(shared_unique_counts)
	plt.pie(np.array(list(shared_unique_counts.values()))[[3, 2, 1, 0, 4]], labels=np.array(list(shared_unique_counts.keys()))[[3, 2, 1, 0, 4]], autopct='%1.1f%%', colors=np.array(colors)[[3, 2, 1, 0, 4]])
	plt.title('Edge Proportions')
	plt.savefig(multimouse_fig_dir + 'pearson corr edge unique shared pie chart.pdf', dpi=500)
	plt.show()

	# # pie chart of edges by shared/unique
	# # by binomial proportions test
	# shared_unique_counts = {'unique_wet':0, 'unique_dry':0, 'shared':0}
	# sig_edge_idx = significant_edge_idx = src.IO.load_pickle('/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/binomtest_sig_edge_indices_precision_0.pkl')
	# for mouseID in mice:
	# 	sig_edges = sig_edge_idx[mouseID]
	# 	mouse_shared_unique_counts = {'unique_wet':0, 'unique_dry':0, 'shared':0}
	# 	drive = src.IO.get_drive(mouseID)
	# 	mouse_dir = drive + '/' + mouseID + '/'
	# 	fig_dir = mouse_dir + f'carry_analysis/figures/'
	# 	days = src.IO.get_carry_days(mouseID)

	# 	calcium_data_path = mouse_dir + days[-1]
	# 	if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
	# 		s2p_fld = calcium_data_path + '/'
	# 	else:
	# 		s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
		
	# 	success_graph = np.load(s2p_fld + f'success_FN_{FN_method}.npy')
	# 	n_neurons = success_graph.shape[0]
	# 	success_weights = success_graph[np.triu_indices(n_neurons, 1)].flatten()
		
	# 	mouse_shared_unique_counts['unique_wet'] = len(sig_edges['wet'])
	# 	mouse_shared_unique_counts['unique_dry'] = len(sig_edges['dry'])
	# 	mouse_shared_unique_counts['shared'] = len([i for i in range(len(success_weights)) if i not in sig_edges['wet'] and i not in sig_edges['dry']])
		
	# 	for key in mouse_shared_unique_counts.keys():
	# 		shared_unique_counts[key] += mouse_shared_unique_counts[key]

	# 	plt.pie(list(mouse_shared_unique_counts.values()), labels=list(mouse_shared_unique_counts.keys()), autopct='%1.1f%%')
	# 	plt.title(f'{mouseID} Edge Proportions')
	# 	plt.savefig(fig_dir + 'pearson corr edge unique shared by binomial prop pie chart.png')
	# 	plt.show()

	# plt.pie(list(shared_unique_counts.values()), labels=list(shared_unique_counts.keys()), autopct='%1.1f%%')
	# plt.title('Edge Proportions')
	# plt.savefig(multimouse_fig_dir + 'pearson corr edge unique shared by binomial prop pie chart.png')
	# plt.show()

