'''
make pie charts!
'''
import matplotlib.pyplot as plt
import numpy as np
import src.utils
import src.IO
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
multimouse_fig_dir = data_dir + '/results/cross_mouse_results/'
plot_cells=False
plot_edges=True
if plot_edges:
	FN_method='pearson_corr'


if plot_cells:
	#keys = ['Increased Firing Rate', 'Decreased Firing Rate', 'Not Informative']
	colors = ['lightcoral', 'firebrick', 'darkred', 'darkgrey']
	load_dir = data_dir + '/results/cross_mouse_results/'
	mod_counts = src.IO.load_pickle(load_dir + 'mod_counts.pkl')
	for m, mouseID in enumerate(mice):
		drive = data_dir + '/neural'
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


if plot_edges:
	# pie chart of edges by shared/unique/sign flip/both zero
	shared_unique_counts = {'active unique':0, 'empty unique':0, 'shared':0, 'sign flip':0, 'null':0}
	s_u_keys = ['active unique', 'empty unique', 'shared', 'sign flip', 'null']
	colors = ['royalblue', 'cornflowerblue', 'rebeccapurple', 'mediumorchid', 'lightgrey']
	for mouseID in mice:
		mouse_shared_unique_counts = {'active unique':0, 'empty unique':0, 'shared':0, 'sign flip':0, 'null':0}
		drive = data_dir + '/neural'
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

