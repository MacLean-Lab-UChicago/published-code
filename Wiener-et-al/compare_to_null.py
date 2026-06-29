'''
compare weights to null distributions
'''
import numpy as np
import src.utils
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore, ttest_ind, false_discovery_control, sem
from collections import Counter
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
FN_method = 'pearson_corr'
correction='bonferroni'
weight=10000

full_pop=True
individual_weights=True
find_sig_weights=True # only runs if individual_weights is True
decodable_neuron_edges=True # only runs if individual_weights is True
find_unique = True # only runs if individual_weights is True
combined_mice= True
show_plots= True

low_weight = f'weighted_PDF_{weight}_'

empty_weights_all_mice = []
active_weights_all_mice = []
sparsity_active = np.zeros(len(mice))
sparsity_empty = np.zeros(len(mice))
decodable_neurons_proportion_edges = []
non_decodable_neurons_proportion_edges = []
decodable_neurons_n_edges = []
non_decodable_neurons_n_edges = []
all_mice_null = {'active':[], 'empty':[]}
all_mice_non_null = {'active':[], 'empty':[]}

for mouse_i, mouseID in enumerate(mice):
	print(mouseID)
	mouse_dir = data_dir + '/neural/' + mouseID + '/'
	days = src.IO.get_carry_days(mouseID)
	s2p_fld = src.IO.get_s2p_fld(mouseID, days[-1])

	null_edges = src.IO.load_pickle(mouse_dir + f'carry_analysis/{low_weight}shuffled_{FN_method}_edge_weights.pkl')
	null_edge_averages = {'active':[], 'empty':[]}
	for cat in ['active', 'empty']:
		if FN_method=='MI':
			null_edge_averages[cat] = np.nanmean(null_edges[cat], axis=1)
		else:
			null_edge_averages[cat] = np.tanh(np.nanmean(np.arctanh(null_edges[cat]), axis=1))

	active_graph = np.load(s2p_fld + f'active_FN_{FN_method}.npy')
	n_neurons = active_graph.shape[0]
	active_weights = active_graph[np.triu_indices(n_neurons, 1)].flatten()
	empty_weights = np.load(s2p_fld + f'empty_FN_{FN_method}.npy')[np.triu_indices(n_neurons, 1)].flatten()

	all_mice_non_null['active'].extend(active_weights)
	all_mice_non_null['empty'].extend(empty_weights)

	all_mice_null['active'].append(null_edges['active'])
	all_mice_null['empty'].append(null_edges['empty'])

	if full_pop:
		if 'corr' in FN_method:
			magnitude_difference = np.tanh(np.mean(np.abs(np.arctanh(active_weights[~np.isnan(active_weights)]))))-np.tanh(np.mean(np.abs(np.arctanh(empty_weights[~np.isnan(empty_weights)]))))
		else:
			magnitude_difference = np.nanmean(active_weights)-np.nanmean(empty_weights)
			non_zero_magnitude_difference = np.nanmean(active_weights[active_weights>0])-np.nanmean(empty_weights[empty_weights>0])

		plt.hist(null_edges['active'].flatten(), color='indianred', alpha=0.5, label='active grasp')
		plt.hist(null_edges['empty'].flatten(), color='k', alpha=0.5, label='empty grasp')
		plt.legend()
		plt.title('Null Distribution Edge Weights')
		plt.savefig(mouse_dir + f'carry_analysis/figures/{mouseID} {low_weight}{FN_method} Null Weights.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

		plt.hist(null_edge_averages['active'].flatten(), color='indianred', alpha=0.5, label='active grasp')
		plt.hist(null_edge_averages['empty'].flatten(), color='k', alpha=0.5, label='empty grasp')
		plt.legend()
		plt.title('Null Distribution Average Edge Weights')
		if show_plots:
			plt.show()
		else:
			plt.close()

		plt.hist(np.abs(null_edge_averages['active'].flatten())-np.abs(null_edge_averages['empty'].flatten()))
		plt.title('Difference Between Null Edge Weight Magnitudes')
		if show_plots:
			plt.show()
		else:
			plt.close()

		if FN_method=='pearson_corr':
			# find mean correlation magnitude for each null model repeat
			# using a z-transform when averaging 
			active_mean_edges = np.tanh(np.nanmean(np.abs(np.arctanh(null_edges['active'])), axis=1))
			empty_mean_edges = np.tanh(np.nanmean(np.abs(np.arctanh(null_edges['empty'])), axis=1))

			# find the effect size comparing data to null distribution of active-empty differences
			plt.hist((active_mean_edges-empty_mean_edges).flatten())
			avg_null_difference = np.mean((active_mean_edges-empty_mean_edges).flatten())
			null_difference_std = np.std((active_mean_edges-empty_mean_edges).flatten())
			print(np.abs(magnitude_difference)-np.abs(avg_null_difference))
			print((magnitude_difference-avg_null_difference)/null_difference_std)
			plt.axvline(magnitude_difference, color='g', label='data')

			# and a p-value as a percentage of the null distribution
			percentile = percentileofscore((active_mean_edges-empty_mean_edges).flatten(), magnitude_difference)
			if percentile>50:
				p = 1-percentile/100
			else:
				p =  percentile/100
			plt.legend()
			plt.title(f'effect={(magnitude_difference-avg_null_difference)/null_difference_std}, p={p}')
			plt.suptitle('Magnitude Difference Between Active Grasp and Empty Correlations')
			plt.savefig(mouse_dir + f'carry_analysis/figures/{mouseID} {FN_method} {low_weight}Null vs Data Correlation Magnitude Differences.png')
			if show_plots:
				plt.show()
			else:
				plt.close()
		else:
			# find mean correlation magnitude for each null model repeat
			active_mean_edges = np.nanmean(null_edges['active'], axis=1)
			empty_mean_edges = np.nanmean(null_edges['empty'], axis=1)

			# find the effect size comparing data to null distribution of active-empty differences
			plt.hist((active_mean_edges-empty_mean_edges).flatten())
			avg_null_difference = np.mean((active_mean_edges-empty_mean_edges).flatten())
			null_difference_std = np.std((active_mean_edges-empty_mean_edges).flatten())
			print(np.abs(magnitude_difference)-np.abs(avg_null_difference))
			print((magnitude_difference-avg_null_difference)/null_difference_std)
			plt.axvline(magnitude_difference, color='g', label='data')
			# and a p-value as a percentage of the null distribution
			percentile = percentileofscore((active_mean_edges-empty_mean_edges).flatten(), magnitude_difference)
			if percentile>50:
				p = 1-percentile/100
			else:
				p =  percentile/100
			plt.legend()
			plt.title(f'effect={(magnitude_difference-avg_null_difference)/null_difference_std}, p={p}')
			plt.suptitle('Difference Between Active Grasp and Empty Weights')
			plt.savefig(mouse_dir + f'carry_analysis/figures/{mouseID} {FN_method} {low_weight}Null vs Data Weight Differences.png')
			if show_plots:
				plt.show()
			else:
				plt.close()

			# get the non-zero edges for each null model repeat
			n_bootstraps = null_edges['active'].shape[0]
			n_edges_active = np.zeros(n_bootstraps)*np.nan
			n_edges_empty = np.zeros(n_bootstraps)*np.nan
			mean_active_non_zero = np.zeros(n_bootstraps)*np.nan
			mean_empty_non_zero = np.zeros(n_bootstraps)*np.nan
			for bootstrap in range(null_edges['active'].shape[0]):
				active_strap = null_edges['active'][bootstrap]
				empty_strap = null_edges['empty'][bootstrap]
				mean_active_non_zero[bootstrap] = np.nanmean(active_strap[active_strap>0])
				mean_empty_non_zero[bootstrap] = np.nanmean(empty_strap[empty_strap>0])
				n_edges_active[bootstrap] = np.sum(active_strap>0)
				n_edges_empty[bootstrap] = np.sum(empty_strap>0)

			# compare non-zero average MI value in the data to the null distribution
			plt.hist((mean_active_non_zero-mean_empty_non_zero).flatten())
			avg_nonz_difference_null = np.mean((mean_active_non_zero-mean_empty_non_zero).flatten())
			plt.axvline(non_zero_magnitude_difference, color='g')
			percentile = percentileofscore((mean_active_non_zero-mean_empty_non_zero).flatten(), non_zero_magnitude_difference)
			if percentile>50:
				p = 1-percentile/100
			else:
				p =  percentile/100
			plt.title(f'effect={np.abs(non_zero_magnitude_difference)-np.abs(avg_nonz_difference_null)}, p={p}')
			plt.suptitle('Difference Between Active Grasp and empty Non-Zero Weights')
			plt.savefig(mouse_dir + f'carry_analysis/figures/{mouseID} {FN_method} {low_weight}Null vs Data Non-Zero Weight Differences.png')
			if show_plots:
				plt.show()
			else:
				plt.close()

	if individual_weights:
		if find_sig_weights:
			sig_weight_indices = {'active':[], 'empty':[]}
			sig_weight_direction = {'active':[], 'empty':[]}
			for cat in ['active', 'empty']:
				if correction=='fdr':
					p_vals = []
					directions=[]
				if cat=='active':
					weights = active_weights
				elif cat=='empty':
					weights = empty_weights
				for i in range(len(weights)):
					null_distr = null_edges[cat][:, i] # grabs all bootstrapped values for that edge
					w = weights[i]
					percentile = percentileofscore(null_distr, w)
					if percentile>50:
						p = 1-percentile/100
						direction = 1 # greater than null distribution
					else:
						p =  percentile/100
						direction = -1 # less than the null distribution
					if correction=='bonferroni':
						if p<(0.05/len(weights)): # Bonferroni correction on p-values
							if FN_method=='MI' and direction==1: # only get edges that are stronger than the null
								sig_weight_indices[cat].append(i)
								sig_weight_direction[cat].append(direction)
							elif direction==1 and w>0: # only get edges that are stronger in magnitude than the null (positive direction)
								sig_weight_indices[cat].append(i)
								sig_weight_direction[cat].append(direction)
							elif direction==-1 and w<0: # only get edges that are stronger in magnitude than the null (negative direction)
								sig_weight_indices[cat].append(i)
								sig_weight_direction[cat].append(direction)
					elif correction=='fdr':
						p_vals.append(p)
						directions.append(direction)
				if correction=='fdr':
					p_transformed = false_discovery_control(np.array(p_vals), method='bh') # Benjamini-Hochberg FDR correction
					if FN_method=='MI':
						sig_weight_indices[cat] = np.where(np.logical_and(p_transformed<0.05, np.array(directions)==1))[0] # only get edges that are significant and stronger than the null
						sig_weight_direction[cat] = np.array(directions)[sig_weight_indices[cat]]
					else:
						sig_weight_indices[cat] = np.where(np.logical_and(p_transformed<0.05, 
							np.logical_or(np.logical_and(np.array(directions)==1, weights>0), np.logical_and(np.array(directions)==-1, weights<0))
							))[0] # only get edges that are significant and stronger magnitude than the null
						sig_weight_direction[cat] = np.array(directions)[sig_weight_indices[cat]]

				print(f'n {cat} edges sig diff from null: ', len(sig_weight_indices[cat]))
				print('prop: ', len(sig_weight_indices[cat])/len(weights))
				print(f'{np.sum(np.array(sig_weight_direction[cat])>0)/len(sig_weight_indices[cat])} more correlated')
				print(f'{np.sum(np.array(sig_weight_direction[cat])<0)/len(sig_weight_indices[cat])} less correlated')
				
				if FN_method=='MI':
					non_zero_edge_idx = np.where(weights>0)[0]
					non_zero_sig_weights = [i for i in non_zero_edge_idx if i in sig_weight_indices[cat]]
					print(f'n {cat} non-zero edges sig diff from null: ', len(non_zero_sig_weights))
					print('prop: ', len(non_zero_sig_weights)/len(non_zero_edge_idx))

				# plot significant edges
				matrix_values = np.zeros(len(weights))
				matrix_values[sig_weight_indices[cat]]=sig_weight_direction[cat]
				matrix = np.zeros((n_neurons, n_neurons))*np.nan
				matrix[np.triu_indices(n_neurons, 1)] = matrix_values
				plt.imshow(matrix)
				plt.title(f'{cat} Edges Significantly Different From Null')
				plt.savefig(mouse_dir + f'carry_analysis/figures/{cat}_sig_edges_comp_to_null_{FN_method}_{low_weight}.png')
				if show_plots:
					plt.show()
				else:
					plt.close()

				plt.hist(weights[sig_weight_indices[cat]].flatten())
				plt.title('Distribution of Weights that Significantly Exist')
				plt.savefig(mouse_dir + f'carry_analysis/figures/{cat}_sig_edges_comp_to_null_{FN_method}_{low_weight}weights.png')
				if show_plots:
					plt.show()
				else:
					plt.close()

			# save the significant edges
			src.IO.save_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_{low_weight}{FN_method}.pkl', sig_weight_indices)
		else:
			sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_{low_weight}{FN_method}.pkl')
		
		if find_unique:
			if FN_method=='pearson_corr':
				unique_shared = {'unique_active':[], 'unique_empty':[], 'shared':[], 'sign_flip':[], 'both_zero':[]}
				# unique active means non-null in active, null in empty
				unique_shared['unique_active'] = np.array([i for i in sig_weight_indices['active'] if i not in sig_weight_indices['empty']], dtype=int)
				# unique active means non-null in empty, null in active
				unique_shared['unique_empty'] = np.array([i for i in sig_weight_indices['empty'] if i not in sig_weight_indices['active']], dtype=int)
				# sign flip means non-null in both active and empty but oppositely signed edge weights
				unique_shared['sign_flip'] = np.array([i for i in sig_weight_indices['active'] if i in sig_weight_indices['empty'] and np.logical_xor(active_weights[i]>0, empty_weights[i]>0)], dtype=int)
				# shared means non-null in both active and empty and the same sign for edge weights
				unique_shared['shared'] = np.array([i for i in sig_weight_indices['active'] if i in sig_weight_indices['empty'] and np.logical_or(np.logical_and(active_weights[i]>0, empty_weights[i]>0), np.logical_and(active_weights[i]<0, empty_weights[i]<0))], dtype=int)
				# both zero means null in both active and empty
				unique_shared['both_zero'] = np.array([i for i in range(len(active_weights)) if i not in sig_weight_indices['empty'] and i not in sig_weight_indices['active']])
				src.IO.save_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl', unique_shared)
			
				print(f"{np.sum(active_weights[unique_shared['sign_flip']]>0)}/{len(unique_shared['sign_flip'])}", np.sum(active_weights[unique_shared['sign_flip']]>0)/len(unique_shared['sign_flip']))
			else:
				unique_shared = {'unique_active':[], 'unique_empty':[], 'shared':[], 'both_zero':[]}
				# unique active means non-null in active, null in empty
				unique_shared['unique_active'] = np.array([i for i in sig_weight_indices['active'] if i not in sig_weight_indices['empty']], dtype=int)
				# unique active means non-null in empty, null in active
				unique_shared['unique_empty'] = np.array([i for i in sig_weight_indices['empty'] if i not in sig_weight_indices['active']], dtype=int)
				# shared means non-null in both active and empty
				unique_shared['shared'] = np.array([i for i in sig_weight_indices['active'] if i in sig_weight_indices['empty']])
				# both zero means null in both active and empty
				unique_shared['both_zero'] = np.array([i for i in range(len(active_weights)) if i not in sig_weight_indices['empty'] and i not in sig_weight_indices['active']])
				src.IO.save_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl', unique_shared)

		active_weights_all_mice.extend(active_weights[sig_weight_indices['active']])
		empty_weights_all_mice.extend(empty_weights[sig_weight_indices['empty']])

		# find the proportion of possible edges that are present
		sparsity_active[mouse_i] = len(sig_weight_indices['active'])/len(active_weights)
		sparsity_empty[mouse_i] = len(sig_weight_indices['empty'])/len(empty_weights)

		# just looking at significantly present edges, is there a magnitude difference between active and empty?
		# plot a histogram of non-null edge weights
		plt.hist(active_weights[sig_weight_indices['active']], color='indianred', alpha=.5, label='active')
		plt.hist(empty_weights[sig_weight_indices['empty']], color='k', alpha=.5, label='empty')
		if FN_method=='pearson_corr':
			# z-transform correlations
			transformed_sig_a = np.arctanh(active_weights[sig_weight_indices['active']])
			transformed_sig_e = np.arctanh(empty_weights[sig_weight_indices['empty']])

			# take the mean and z-transform back
			mean_a = np.tanh(np.nanmean(transformed_sig_a))
			mean_e = np.tanh(np.nanmean(transformed_sig_e))

			# independent t-test for difference between active and empty mean edge weight
			result = ttest_ind(transformed_sig_a, transformed_sig_e)
			pval = result.pvalue

			plt.axvline(mean_a, color='indianred')
			plt.axvline(mean_e, color='k')
			plt.title(f'p={pval}')
			plt.suptitle('Histogram of Significant Edge Weights')
			plt.savefig(mouse_dir + f'carry_analysis/figures/comparison_of_sig_edges_comp_to_null_{FN_method}_{low_weight}weights.png')
			if show_plots:
				plt.show()
			else:
				plt.close()

			# plot a histogram of non-null edge weight magnitudes
			plt.hist(np.abs(active_weights[sig_weight_indices['active']]), color='indianred', alpha=.5, label='active')
			plt.hist(np.abs(empty_weights[sig_weight_indices['empty']]), color='k', alpha=.5, label='empty')

			# take the average of z-transformed weight magnitudes and z-transform back
			mean_a = np.tanh(np.nanmean(np.abs(transformed_sig_s)))
			mean_e = np.tanh(np.nanmean(np.abs(transformed_sig_d)))

			# independent t-test for difference between active and empty mean edge magnitude
			result = ttest_ind(np.abs(transformed_sig_s), np.abs(transformed_sig_d))
			pval = result.pvalue

			plt.axvline(mean_a, color='indianred')
			plt.axvline(mean_e, color='k')
			plt.title(f'p={pval}')
			plt.suptitle('Histogram of Significant Edge Weight Magnitudes')
			plt.savefig(mouse_dir + f'carry_analysis/figures/comparison_of_sig_edges_comp_to_null_{FN_method}_{low_weight}_weight_magitudes.png')
			if show_plots:
				plt.show()
			else:
				plt.close()

		else:
			# take the mean over all non-null weights
			mean_a = np.nanmean(active_weights[sig_weight_indices['active']])
			mean_e = np.nanmean(empty_weights[sig_weight_indices['empty']])

			# independent t-test for difference between active and empty mean edge weight
			result = ttest_ind(active_weights[sig_weight_indices['active']], empty_weights[sig_weight_indices['empty']])
			pval = result.pvalue

			plt.axvline(mean_a, color='indianred')
			plt.axvline(mean_e, color='k')
			plt.title(f'p={pval}')
			plt.suptitle('Histogram of Significant Edge Weights')
			plt.savefig(mouse_dir + f'carry_analysis/figures/comparison_of_sig_edges_comp_to_null_{FN_method}_{low_weight}weights.png')
			if show_plots:
				plt.show()
			else:
				plt.close()

		print()

		if decodable_neuron_edges:
			# load in decodable neurons
			decodable_neurons = np.load(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy')
			# adjust indices for NaN cells
			NaN_cells = np.where(np.load(s2p_fld + 'NaN_containing_cells_bool.npy'))[0]
			if len(NaN_cells)>0:
				for NaN_cell in NaN_cells:
					decodable_neurons[decodable_neurons>NaN_cell]-=1
			
			indices = np.triu_indices(active_graph.shape[0], 1)
			n_neurons = active_graph.shape[0]
			for trial_type in ['active', 'empty']:
				sig_boolean = np.zeros_like(active_weights, dtype=bool)
				sig_boolean[sig_weight_indices[trial_type]]=True
				graphlike_sig_edges = np.zeros(active_graph.shape, dtype=bool)
				graphlike_sig_edges[indices] = sig_boolean
				cell_idx_for_sig_edges = np.where(graphlike_sig_edges)

				print(mouseID)
				print(trial_type)
				print(FN_method)
				sig_cell_sig_edge = [cell for cell in decodable_neurons if cell in cell_idx_for_sig_edges[0] or cell in cell_idx_for_sig_edges[1]]
				print('prop decodable neurons w/ sig edges: ', len(sig_cell_sig_edge)/len(decodable_neurons), f'{len(sig_cell_sig_edge)}/{len(decodable_neurons)}')
				
				any_cell_sig_edge = [cell for cell in range(active_graph.shape[0]) if cell in cell_idx_for_sig_edges[0] or cell in cell_idx_for_sig_edges[1]]
				print('prop neurons w/ sig edges: ', len(any_cell_sig_edge)/active_graph.shape[0], f'{len(any_cell_sig_edge)}/{active_graph.shape[0]}')
				
				neurons_in_sig_edges = np.append(cell_idx_for_sig_edges[0], cell_idx_for_sig_edges[1])
				edge_neuron_counts = Counter(neurons_in_sig_edges)
				edge_count_decodable_neurons = np.array([edge_neuron_counts[neuron] for neuron in decodable_neurons])
				edge_count_nondecodable_neurons = np.array([edge_neuron_counts[neuron] for neuron in range(n_neurons) if neuron not in decodable_neurons])
				decodable_neurons_n_edges.extend(edge_count_decodable_neurons)
				non_decodable_neurons_n_edges.extend(edge_count_nondecodable_neurons)

				# find proportion of neurons decodable/non-decodable has edges with
				possible_edges_per_neuron = n_neurons-1
				decodable_prop_edges = edge_count_decodable_neurons/possible_edges_per_neuron
				nondecodable_prop_edges = edge_count_nondecodable_neurons/possible_edges_per_neuron
				decodable_neurons_proportion_edges.extend(decodable_prop_edges)
				non_decodable_neurons_proportion_edges.extend(nondecodable_prop_edges)

				print('proportion edges for decodable neurons: ', np.mean(decodable_prop_edges))
				print('proportion edges for non-decodable neurons: ', np.mean(nondecodable_prop_edges))

				avg_edge_count_all_neurons = np.mean(np.array([edge_neuron_counts[a] for a in range(n_neurons)]))
				avg_edge_count_decodable_neurons = np.mean(edge_count_decodable_neurons)
				avg_edge_count_nondecodable_cells = np.mean(edge_count_nondecodable_neurons)
				print('avg number of edges for decodable neurons: ', avg_edge_count_decodable_neurons)
				print('avg number of edges for all neurons: ', avg_edge_count_all_neurons)
				print('avg number of edges for non-decodable neurons: ', avg_edge_count_nondecodable_cells)

				sig_edge_if_one_sig_cell = [edge for edge in range(len(sig_weight_indices[trial_type])) if cell_idx_for_sig_edges[0][edge] in decodable_neurons or cell_idx_for_sig_edges[1][edge] in decodable_neurons]
				print('prop edges w decodable neurons involved: ', len(sig_edge_if_one_sig_cell)/len(sig_weight_indices[trial_type]), f'{len(sig_edge_if_one_sig_cell)}/{len(sig_weight_indices[trial_type])}')
				
				sig_edge_if_both_sig_cell = [edge for edge in range(len(sig_weight_indices[trial_type])) if cell_idx_for_sig_edges[0][edge] in decodable_neurons and cell_idx_for_sig_edges[1][edge] in decodable_neurons]
				print('prop edges w two decodable neurons involved: ', len(sig_edge_if_both_sig_cell)/len(sig_weight_indices[trial_type]), f'{len(sig_edge_if_both_sig_cell)}/{len(sig_weight_indices[trial_type])}')
				print()

	if combined_mice:
		sig_weight_indices = src.IO.load_pickle(mouse_dir + f'carry_analysis/sig_edges_comp_to_null_{low_weight}{FN_method}.pkl')
		active_weights_all_mice.extend(active_weights[sig_weight_indices['active']])
		empty_weights_all_mice.extend(empty_weights[sig_weight_indices['empty']])


if combined_mice:
	multi_mouse_fig_dir = data_dir + '/results/cross_mouse_results/'

	if individual_weights:
		print('empty sparsity: ', sparsity_empty, np.mean(sparsity_empty), '+-', sem(sparsity_empty))
		print('active sparsity: ', sparsity_active, np.mean(sparsity_active), '+-', sem(sparsity_active))

	active_weights_all_mice = np.array(active_weights_all_mice)
	empty_weights_all_mice = np.array(empty_weights_all_mice)

	all_mice_null['active'] = np.concatenate(all_mice_null['active'], axis=1)
	all_mice_null['empty'] = np.concatenate(all_mice_null['empty'], axis=1) 
	all_mice_non_null['active'] = np.array(all_mice_non_null['active'])
	all_mice_non_null['empty'] = np.array(all_mice_non_null['empty']) 
	
	if FN_method=='pearson_corr':
		# z-transform the correlations
		transformed_a = np.arctanh(active_weights_all_mice)
		transformed_e = np.arctanh(empty_weights_all_mice)

		# take the mean separately for positive and negative correlations
		mean_pos_a = np.tanh(np.nanmean(transformed_a[transformed_a>0]))
		mean_pos_e = np.tanh(np.nanmean(transformed_e[transformed_e>0]))
		mean_neg_a = np.tanh(np.nanmean(transformed_a[transformed_a<0]))
		mean_neg_e = np.tanh(np.nanmean(transformed_e[transformed_e<0]))

		# independent t-test for active/empty grasp weight differences on z-transformed correlations (only non-null edges)
		# separately for positive and negative correlations
		# both testing that active is stronger in magnitude (thus greater for + and less for -)
		pos_result = ttest_ind(transformed_a[transformed_a>0], transformed_e[transformed_e>0], alternative='greater')
		neg_result = ttest_ind(transformed_a[transformed_a<0], transformed_e[transformed_e<0], alternative='less')
		print(pos_result)
		print(neg_result)

		# plot correlation weight distributions and means
		plt.hist(active_weights_all_mice, bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='indianred', alpha=0.4, label='active grasp')
		plt.hist(empty_weights_all_mice, bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='k', alpha=0.4, label='empty grasp')
		plt.xlabel('Pearson Correlation')
		plt.ylabel('Edge Count')
		plt.axvline(mean_neg_e, color='k')
		plt.axvline(mean_pos_e, color='k')
		plt.axvline(mean_pos_a, color='indianred')
		plt.axvline(mean_neg_a, color='indianred')
		plt.legend()
		plt.title(f'- weights, p={neg_result.pvalue:.4f}, + weights, p={pos_result.pvalue:.4f}')
		print(mean_neg_a-mean_neg_e, neg_result.pvalue, mean_pos_a-mean_pos_e, pos_result.pvalue)
		plt.suptitle('Edge Weights in Active vs Empty Grasp')
		#plt.savefig(f'{multi_mouse_fig_dir}Pearson Correlation weight differences.pdf', dpi=550)
		if show_plots:
			plt.show()
		else:
			plt.close()

		# take the mean magnitude
		mean_abs_w = np.tanh(np.nanmean(np.abs(transformed_a)))
		mean_abs_d = np.tanh(np.nanmean(np.abs(transformed_e)))

		# independent t-test for active/empty grasp weight differences on z-transformed correlation magnitudes (only non-null edges)
		result = ttest_ind(np.abs(transformed_a), np.abs(transformed_e), alternative='greater')
		print(result)

		# plot correlation magnitude distributions and means
		fig, ax = plt.subplots()
		ax.hist(np.abs(active_weights_all_mice), bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='indianred', alpha=0.4, label='active grasp')
		ax.hist(np.abs(empty_weights_all_mice), bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='k', alpha=0.4, label='empty grasp')
		ax.set_xlabel('Pearson Correlation Magnitude')
		ax.set_ylabel('Functional Connection Count')
		ax.legend()
		ax.axvline(mean_abs_d, color='k')
		ax.axvline(mean_abs_w, color='indianred')
		ax.set_title(f'p={result.pvalue:.4f}')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		print(mean_abs_w-mean_abs_d, result.pvalue)
		#plt.suptitle('Correlation Magnitudes')
		#plt.savefig(f'{multi_mouse_fig_dir}Pearson Correlation weight magnitude differences.pdf', dpi=500)
		if show_plots:
			plt.show()
		else:
			plt.close()

		# plot positive and negative FCs separately

		# use mann-whitney (Wilcoxon signed-rank to compare medians instead of means)
		pos_result = mannwhitneyu(active_weights_all_mice[active_weights_all_mice>0], empty_weights_all_mice[empty_weights_all_mice>0], alternative='greater')
		print(pos_result)

		# plot positive FCs and medians
		fig, ax = plt.subplots()
		ax.hist(active_weights_all_mice[active_weights_all_mice>0], bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='indianred', alpha=0.4, label='active grasp')
		ax.hist(empty_weights_all_mice[empty_weights_all_mice>0], bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='k', alpha=0.4, label='empty grasp')
		ax.set_xlabel('Pearson Correlation')
		ax.set_ylabel('Edge Count')
		ax.axvline(np.median(active_weights_all_mice[active_weights_all_mice>0]), color='indianred')
		ax.axvline(np.median(empty_weights_all_mice[empty_weights_all_mice>0]), color='k')
		ax.legend()
		ax.set_title(f'+ weights, p={pos_result.pvalue:.4f}')
		#print(mean_neg_w-mean_neg_d, neg_result.pvalue, mean_pos_w-mean_pos_d, pos_result.pvalue)
		fig.suptitle('Positive Edge Weights in Active vs Empty Grasp')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		plt.savefig(f'{multi_mouse_fig_dir}Positive Pearson Correlation weight differences.pdf', dpi=550)
		if show_plots:
			plt.show()
		else:
			plt.close()

		# repeat for negative FCs
		neg_result = mannwhitneyu(active_weights_all_mice[active_weights_all_mice<0], empty_weights_all_mice[empty_weights_all_mice<0], alternative='less')
		print(neg_result)

		fig, ax = plt.subplots()
		ax.hist(active_weights_all_mice[active_weights_all_mice<0], bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), 0), color='indianred', alpha=0.4, label='active grasp')
		ax.hist(empty_weights_all_mice[empty_weights_all_mice<0], bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), 0), color='k', alpha=0.4, label='empty grasp')
		ax.set_xlabel('Pearson Correlation')
		ax.set_ylabel('Edge Count')
		ax.axvline(np.median(active_weights_all_mice[active_weights_all_mice<0]), color='indianred')
		ax.axvline(np.median(empty_weights_all_mice[empty_weights_all_mice<0]), color='k')
		ax.legend()
		ax.set_title(f'- weights, p={neg_result.pvalue:.4f}')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		#print(mean_neg_w-mean_neg_d, neg_result.pvalue, mean_pos_w-mean_pos_d, pos_result.pvalue)
		fig.suptitle('Negative Edge Weights in Active vs Empty Grasp')
		plt.savefig(f'{multi_mouse_fig_dir}Negative Pearson Correlation weight differences.pdf', dpi=550)
		if show_plots:
			plt.show()
		else:
			plt.close()

		# plot FC magnitudes (absolute value of weights) and compare medians

		# find FC magnitude medians
		median_abs_w = np.median(np.abs(active_weights_all_mice))
		median_abs_d = np.median(np.abs(empty_weights_all_mice))

		# Wilcoxon rank sum test to compare magnitude medians
		mann_u_res = mannwhitneyu(np.abs(active_weights_all_mice), np.abs(empty_weights_all_mice), alternative='greater')
		print(mann_u_res)
		print(median_abs_d, median_abs_w)

		# plot magntiude distributions and medians
		fig, ax = plt.subplots()
		ax.hist(np.abs(active_weights_all_mice), bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='indianred', alpha=0.4, label='active grasp')
		ax.hist(np.abs(empty_weights_all_mice), bins=20, range = (0, np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='k', alpha=0.4, label='empty grasp')
		ax.set_xlabel('Pearson Correlation Magnitude')
		ax.set_ylabel('Functional Connection Count')
		ax.legend()
		ax.axvline(median_abs_d, color='k')
		ax.axvline(median_abs_w, color='indianred')
		ax.set_title(f'p={mann_u_res.pvalue:.4f}')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		#print(mean_abs_w-mean_abs_d, result.pvalue)
		print(median_abs_w-median_abs_d, mann_u_res.pvalue)
		#plt.suptitle('Correlation Magnitudes')
		plt.savefig(f'{multi_mouse_fig_dir}Pearson Correlation weight magnitude differences.pdf', dpi=500)
		if show_plots:
			plt.show()
		else:
			plt.close()

	else: # if Mutual Information as the FN method
		# find the mean weight
		median_a = np.nanmedian(active_weights_all_mice)
		median_e = np.nanmedian(empty_weights_all_mice)

		# Wilcoxon rank-sum to see if active has higher information weights than empty
		result = mannwhitneyu(active_weights_all_mice, empty_weights_all_mice, alternative='greater')

		# plot a histogram of all edges and means
		plt.hist(active_weights_all_mice, bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='indianred', alpha=0.4, label='active grasp')
		plt.hist(empty_weights_all_mice, bins=20, range = (np.min(np.append(active_weights_all_mice, empty_weights_all_mice)), np.max(np.append(active_weights_all_mice, empty_weights_all_mice))), color='k', alpha=0.4, label='empty grasp')
		plt.xlabel('Mutual Information')
		plt.ylabel('Functional Connection Count')
		plt.legend()
		plt.axvline(median_e, color='k')
		plt.axvline(median_a, color='indianred')
		plt.title(f'p={result.pvalue:.4f}')
		plt.suptitle('Edge Weights in Active vs Empty Grasp')
		plt.savefig(f'{multi_mouse_fig_dir}MI weight differences')
		if show_plots:
			plt.show()
		else:
			plt.close()

	# plot null distribution edge weight distributions
	plt.hist(np.abs(all_mice_null['active']).flatten(), bins=25, range=(0, 1), color='indianred', alpha=0.5, label='active grasp, null')
	plt.hist(np.abs(all_mice_null['empty']).flatten(), bins=25, range=(0, 1), color='k', alpha=0.5, label='empty grasp, null')
	plt.legend()
	plt.title('Null Distribution Edge Weights')
	plt.savefig(multi_mouse_fig_dir + 'Null Average Magnitude Distributions all mice.png')
	if show_plots:
		plt.show()
	else:
		plt.close()

	# compare weight magnitude chages to the null model
	if FN_method=='pearson_corr':
		# find the average weight magnitude difference between active and empty per null model boostrap (with a z-transform)
		# this is then the magnitude weight difference that can be purely attributed to firing rate and spatiotemporal timing differences
		active_mean_edges = np.tanh(np.nanmean(np.abs(np.arctanh(all_mice_null['active'])), axis=1))
		empty_mean_edges = np.tanh(np.nanmean(np.abs(np.arctanh(all_mice_null['empty'])), axis=1))

		# plot the null distribution of weight magnitude differences
		plt.hist((active_mean_edges-empty_mean_edges).flatten(), bins=10)
		plt.title('Difference Between Null Edge Weight Magnitudes')
		plt.savefig(multi_mouse_fig_dir + 'Null Average Correlation Magnitude Difference Distribution all mice.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

		# find the mean weight magnitude difference in the actual data
		magnitude_difference = np.tanh(np.nanmean(np.arctanh(np.abs(all_mice_non_null['active'])-np.abs(all_mice_non_null['empty']))))

		# plot the null distribution compared to the data
		plt.hist((active_mean_edges-empty_mean_edges).flatten(), bins=10, color='darkgrey', label='Null Model')
		avg_null_difference = np.mean((active_mean_edges-empty_mean_edges).flatten())
		std_null_difference = np.std((active_mean_edges-empty_mean_edges).flatten())
		print(np.abs(magnitude_difference)-np.abs(avg_null_difference))
		print((magnitude_difference-avg_null_difference)/std_null_difference)
		plt.axvline(magnitude_difference, color='indianred', label='Data')
		percentile = percentileofscore((active_mean_edges-empty_mean_edges).flatten(), magnitude_difference)
		if percentile>50:
			p = 1-percentile/100
		else:
			p =  percentile/100
		plt.legend()
		plt.title(f'effect={(magnitude_difference-avg_null_difference)/std_null_difference}, p={p}')
		plt.suptitle('Magnitude Difference Between Active Grasp and Empty Correlations Vs Null')
		plt.savefig(multi_mouse_fig_dir + 'Null vs Data Correlation Magnitude Differences.pdf', dpi=550)
		if show_plots:
			plt.show()
		else:
			plt.close()
	else:
		# find the average weight difference between active and empty per null model boostrap (with a z-transform)
		# this is then the weight difference that can be purely attributed to firing rate and spatiotemporal timing differences
		active_mean_edges = np.nanmean(all_mice_null['active'], axis=1)
		empty_mean_edges = np.nanmean(all_mice_null['empty'], axis=1)

		# plot the null distribution of weight differences (also magnitude because MI is only positive)
		plt.hist((active_mean_edges-empty_mean_edges).flatten(), bins=10)
		plt.title('Difference Between Null Edge Weight Magnitudes')
		plt.savefig(multi_mouse_fig_dir + 'Null Average Correlation Magnitude Difference Distribution all mice.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

		# find the weight differences in the actual data
		active_mean = np.nanmean(all_mice_non_null['active'])
		empty_mean = np.nanmean(all_mice_non_null['empty'])
		magnitude_difference = active_mean-empty_mean

		# plot null distribution compared to data weight differences
		plt.hist((active_mean_edges-empty_mean_edges).flatten(), color='darkgrey', label='Null Model')
		avg_null_difference = np.mean((active_mean_edges-empty_mean_edges).flatten())
		print(np.abs(magnitude_difference)-np.abs(avg_null_difference))
		print((magnitude_difference-avg_null_difference)/std_null_difference)
		plt.axvline(magnitude_difference, color='indianred', label='Data')
		percentile = percentileofscore((active_mean_edges-empty_mean_edges).flatten(), magnitude_difference)
		if percentile>50:
			p = 1-percentile/100
		else:
			p =  percentile/100
		plt.legend()
		plt.title(f'effect={(magnitude_difference-avg_null_difference)/std_null_difference}, p={p}')
		plt.suptitle('Difference Between Active Grasp and Empty Weights')
		plt.savefig(multi_mouse_fig_dir + 'Null vs Data MI Differences.png')
		if show_plots:
			plt.show()
		else:
			plt.close()

