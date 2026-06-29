'''
visualize sorted functional networks
'''
import numpy as np
import matplotlib.pyplot as plt
import src.IO
import scipy

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
FN_method = 'pearson_corr'
show_figs=True

for mouseID in mice:
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	save_info_dir = mouse_dir + 'carry_analysis/'
	save_dir = mouse_dir + 'carry_analysis/figures/'

	days = src.IO.get_carry_days(mouseID)
	s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])
	
	active_graph = np.load(s2p_fld + f'success_FN_{FN_method}.npy')
	empty_graph = np.load(s2p_fld + f'dry_FN_{FN_method}.npy')
	n_neurons = active_graph.shape[0]
	if FN_method=='MI':
		indexing = scipy.io.loadmat(s2p_fld + 'FN_sort_MI.mat')
	else:
		indexing = scipy.io.loadmat(s2p_fld + 'FN_sort_pearson.mat')

	idx = indexing['indsort'].flatten()-1

	
	vmax = np.percentile(np.concatenate([active_graph[np.triu_indices(active_graph.shape[0], k=1)], empty_graph[np.triu_indices(active_graph.shape[0], k=1)]]), 99.9)
	print(vmax)

	if FN_method=='MI':
		vmin = 0
	else:
		vmin=-vmax

	if FN_method=='MI':
		plt.imshow(active_graph[idx, :][:, idx], cmap='Blues', vmin=vmin, vmax=vmax, rasterized=True)
	else:
		plt.imshow(active_graph[idx, :][:, idx], cmap='RdBu', vmin=vmin, vmax=vmax, rasterized=True)
	plt.title(f'Active Grasp Functional Network')
	plt.xlabel('Neurons')
	plt.ylabel('Neurons')
	plt.colorbar()
	plt.savefig(save_dir + f'{mouseID} Active Grasp FN {FN_method} community sorted.pdf', dpi=500)
	if show_figs:
		plt.show()
	else:
		plt.close()

	if FN_method=='MI':
		plt.imshow(empty_graph[idx, :][:, idx], cmap='Blues', vmin=vmin, vmax=vmax, rasterized=True)
	else:
		plt.imshow(empty_graph[idx, :][:, idx], cmap='RdBu', vmin=vmin, vmax=vmax, rasterized=True)
	plt.title(f'Empty Grasp Functional Network')
	plt.colorbar()
	plt.xlabel('Neurons')
	plt.ylabel('Neurons')
	plt.savefig(save_dir + f'{mouseID} Empty Grasp FN {FN_method} community sorted.pdf', dpi=500)
	if show_figs:
		plt.show()
	else:
		plt.close()

	difference = active_graph-empty_graph
	if FN_method=='MI':
		vmin=-vmax

	plt.imshow(difference[idx, :][:, idx], cmap='RdGy_r', vmin=vmin, vmax=vmax, rasterized=True)
	plt.title(f'Active-Empty Functional Network')
	plt.colorbar()
	plt.xlabel('Neurons')
	plt.ylabel('Neurons')
	plt.savefig(save_dir + f'{mouseID} Active vs Empty Grasp FN {FN_method} community sorted.pdf', dpi=500)
	if show_figs:
		plt.show()
	else:
		plt.close()