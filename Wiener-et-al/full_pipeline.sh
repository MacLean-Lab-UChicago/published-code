#!/bin/bash

# abbreviations used throughout:
#	Functional Connection (FC)
#	Functional Network (FN)
# 	Mutual Information (MI)

# TO DO:
# 	- for all scripts: make paths relative and add a data folder containing the base data needed to run stuff (CellReg, cascade_spks, carry_times, carry_labels, kinematic information)


# FIGURE 1
# plot carry trajectories
python plot_carry_trajectories.py

# run kinematics SVM and plot
python kinematics_svm.py
# requires (per mouse):
#	paw_centroid_interpolated, aperature_interpolated, d2d3_splay_interpolated, relative_y_normal_to_palm_interpolated
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs:
# 	{mouseID}_kinematics_svm_accuracies_test_.1.npy, {mouseID}_gross_kinematics_svm_accuracies_test_.1.npy, {mouseID}_fine_kinematics_svm_accuracies_test_.1.npy 
#		(numpy array of all test accuracies for each mouse for full, gross, and fine kinematics svm decoders at test size .1)
#	gross_SVM_accuracies, gross_SVM_error, gross_95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for gross kinematics svm decoders)
#	fine_SVM_accuracies, fine_SVM_error, fine_95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for fine kinematics svm decoders)
#	SVM_accuracies, SVM_error, 95_CI (numpy arrays of accuracies, error, and 95% CI for each mouse for full kinematics svm decoders)
#	mouse-averaged kinematic svm accuracy.pdf (bar plot of kinematic SVM accuracies, separated by gross, fine, and full)


# FIGURE 2 AND RELATED SUPPLEMENTS
# plot active and empty counts and proportions over time
python learning_plot_trial_counts.py
# requires (for each mouse and each day of training):
#	{mouse}_{day}_classified_pellet_presence_updated.pkl (pickled dictionary of carry classifications per mouse per training day)
# outputs:
#	all mice active empty carry counts by type over time.pdf (line plot (+- standard error) of the number of carry trials by type over training)
#	all mice active empty carry proportions by type over time.pdf (line plot (+- standard error) of the proportion of carry trials by type over training)

# plot peak and crossover day distributions
python plot_empty_peak_and_crossover_distribution.py
# no requirements
# outputs:
# 	peak and crossover timeline.pdf (timeline-style plot with empty peak and active-empty crossovers day distributions across mice)

# plot success rate
python learning_plot_success_rate.py
# requires (for each mouse):
#	success_fail_bool.npy (boolean array containing manual success/fail labels for each grasp and carry for each training day)
#	{mouseID}_{day}_pellet_frames_and_presence_dict.pkl (pickled dictionary with pedestal shifts and if a pellet was present, separated by event per recording session)
# outputs:
#	{mouseID} number of successes.png (plot of success count per mouse)
#	{mouseID} success rate.png (plot of success rate per mouse)
#	number of successes.pdf (plot of success counts, average +- sem across mice per day)
#	success rate.pdf (plot of success rate, average +- sem across mice per day)
#	success rate and count overlayed.pdf (overlayed plot of success rate and counts, average +- sem across mice per day)


# FIGURE 3 AND RELATED SUPPLEMENTS

# this series of scripts:
# finds decodable cells
# finds increased, decreased, and time-dependent decodable cells
# does decoding from population-level neural activity and kinematics
# plots bar graphs of decoding accuracies
# plots a piechart of decodable cells by type (increased, decreased, time-dependent, not decodable)
python modulated_via_avgfr_svm.py && python modulated_via_svm.py && \
python compare_mod_methods.py && \
python population_decoder_svm.py && \
python plot_svm_accuracy.py && \
python plot_decodable_cells_piechart.py
# first:run per cell SVM average and SVM instant firing rate
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs (for each mouse): 
#	SVM_avgfr_active_grasp_modulated_cells (numpy boolean array of decodable cells in average firing rate model)
#	single_cell_SVM_avgfr_accuracies_by_cell (numpy array of all cell decoding accuracy in average firing rate model)
#	single_cell_SVM_avgfr_confidence_intervals_by_cell(numpy array of decoding 95% CI in average firing rate model)
#	SVM_active_grasp_modulated_cells (numpy boolean array of decodable cells in instant firing rate model)
#	single_cell_SVM_accuracies_by_cell (numpy array of all cell decoding accuracy in instant firing rate model)
#	single_cell_SVM_confidence_intervals_by_cell(numpy array of decoding 95% CI in instant firing rate model)

# then combine the two decoder results
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# 	SVM_avgfr_active_grasp_modulated_cells
# 	SVM_active_grasp_modulated_cells
# outputs (for each mouse): 
#	SVM_combined_active_grasp_modulated_cell_indices (numpy array of decodable cell indices)
#	SVM_time_averaged_active_grasp_modulated_cell_indices (numpy array of non-time-dependent decodable cell indices)
#	SVM_time_varying_active_grasp_modulated_cell_indices (numpy array of time-dependent decodable cell indices)
# singlular output:
#	mod_counts (pickle file containing a dictionary of increased, decreased, time-varying, and non-decodable cell counts by mouse)

# then run full population SVM
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	SVM_combined_active_grasp_modulated_cell_indices
# outputs: 
#	SVM_accuracies_test_.1, SVM_error_test_.1, 95_CI_test_.1 (numpy arrays of accuracies, error, and 95% CI for each mouse for full families neural svm decoders at test size .1)
#	Control_SVM_accuracies_test_.1, Control_SVM_error_test_.1, Control_95_CI_test_.1 (numpy arrays of accuracy, errors, and 95% CI for each mouse for neuron count-matched control models)

# plot the SVM accuracies
# requires: 
#	SVM_accuracies_test_.1, SVM_error_test_.1, 95_CI_test_.1
#	SVM_accuracies, gross_SVM_accuracies, fine_SVM_accuracies (from kinematics_svm.py)
# outputs: 	
#	mouse_avg_kin_vs_neural_avg_fr.pdf (barplot of mouse-averaged neural compared to kinematic decoding performance by model)
#	neural_pop_decoding_avg_fr.pdf (barplot of mouse-averaged decoding by the three neural populations - non-decodable, decodable, and full - and kinematics)

# plot the decodable cell percentages as a piechart
# requires: 
#	mod_counts (pickle file containing a dictionary of increased, decreased, time-varying, and non-decodable cell counts by mouse)
# outputs: 	
#	mod cells pie chart.png (piechart of % decodable cells by category by mouse)
#	mod cells pie chart.pdf (piechart of % decodable cells by category across all mice)


# plot individual cell PETHs during active and empty
python PETH_by_cell.py
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	SVM_combined_active_grasp_modulated_cell_indices.npy
# outputs (per mouse):
#	Trial Firing Rates By Condition y log scale.png (histogram of single trial average firing rates per neuron by condition, y-axis on log scale)
#	Trial Firing Rates By Condition.png (histogram of single trial average firing rates per neuron by condition)
#	population_averaged_PETH.png (line plot of population-averaged PETH for active and empty for the neurons of a single mouse)
# outputs (per decodable cell):
#	Cell {cell} PETH.pdf (line plot of single cell PETH for active and empty)
# outputs (singular):
#	population_averaged_PETH.pdf (line plot of population-averaged PETH for active and empty across all neurons)

# find mutual information between neurons and trial category
# then plot that mutual information
python jidt_MI.py && python examine_node_MI.py
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	SVM_combined_active_grasp_modulated_cell_indices.npy
# outputs (per mouse):
#   node_trialcat_MI.npy
#	node_trialcat_MI_std.npy
#	node_trialcat_MI_continous_assumption.npy
#	node_trialcat_MI_std_continous_assumption.npy

# plot population PETHs
python carry_PETHs.py
# requires (per mouse):
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs (per mouse):
#	carry_{class_to_sort_by}_sort_indices.npy (array of sort indices for each carry condition)
#	active_empty_carry_PETHs_norm.png (normalized tiled PETH. 2x2 grid where rows are sorted by active, then empty and columns are active and empty activity)

# find correlation between active and empty trials in the neural activity
python compare_PETH_correlations.py
# requires (per mouse):
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs (per mouse):
#	wi vs x-category PETH correlations.png (violin plot of within- vs across-category PETH correlations and CellID shuffle correlations)
# outputs (singular):
#	wi vs x-category PETH half correlations.png (violin plot of within- vs across-category PETH scorrelations and CellID shuffle correlations, all cells)

python single_trial_autocorrelations.py 
# requires (per mouse):
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs (per mouse):
#	single trial autocorrelations all comparisons.png (violin plot of all pairwise trial correlations, split into within- and across-categories)
#	single trial autocorrelations averaged by cell.png (violin plot of pairwise trial correlations, split into within- and across-categories, averaged by cell)
# outputs (singular):
#	single trial autocorrelations averaged by cell.png (violin plot of pairwise trial correlations, split into within- and across-categories, averaged by cell, all cells)


# FIGURE 4 AND RELATED SUPPLEMENTS

# this series of scripts:
# calculates the functional networks and null distributions
# then finds the significant and null functional connections based on comparing to the null distribution
# organizes FNs for visualization by community 
# and plots them
python calculate_FNs.py && python null_edge_distributions.py && \
python compare_to_null.py && \
matlab -batch "FN_visualization" && \
python visualize_FNs.py

# first, calculate functional networks and null model (python calculate_FNs.py && python null_edge_distributions.py)
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
# outputs (for each mouse):
#	NaN_containing_cells_bool (boolean numpy array of which cells contain NaNs during relevant trials)
#	active_FN_pearson_corr; empty_FN_pearson_corr (numpy array functional networks per mouse for active (success) and empty (dry))
#	weighted_PDF_{weight}_shuffled_{FN_method}_edge_weights (pickle file containing dictionary by trial category of null distribution FN weights)	
#	weighted_PDF_{weight}_shuffled_{FN_method}_averaged_edge_weights (pickle file containing dictionary by trial category of average null distribution FN weights by pairwise relationship)

# then get significant edges by comparing to the null distributions (python compare_to_null.py)
# requires (for each mouse):
#	active_FN_pearson_corr; empty_FN_pearson_corr (numpy array functional networks per mouse for active (success) and empty (dry))
#	weighted_PDF_{weight}_shuffled_{FN_method}_edge_weights (pickle file containing dictionary by trial category of null distribution FN weights)	
#	NaN_containing_cells_bool (boolean numpy array of which cells contain NaNs during relevant trials)
#	SVM_combined_active_grasp_modulated_cell_indices (numpy array of decodable cell indices)
# outputs (for each mouse):
# 	sig_edges_comp_to_null_{low_weight}{FN_method}.pkl (pickle file containing dictionary by trial category of non-null FC indices)
# 	unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl' (pickle file containing dictionary with unique/shared classifications of FC indices)
# 	{mouseID} {low_weight}{FN_method} Null Weights.png (histogram of the null model distribution of FC weights)
#	{mouseID} {FN_method} {low_weight}Null vs Data Correlation Magnitude Differences.png (null model correlation magnitude difference between active and empty compared to data)
#	{cat}_sig_edges_comp_to_null_{FN_method}_{low_weight}.png (matrix showing the location of non-null correlations; for active and empty)
#	{cat}_sig_edges_comp_to_null_{FN_method}_{low_weight}weights.png (histogram of non-null FC weights)
#	comparison_of_sig_edges_comp_to_null_{FN_method}_{low_weight}weights.pdf (histogram of active vs empty non-null weights)
#	comparison_of_sig_edges_comp_to_null_{FN_method}_{low_weight}_weight_magitudes.pdf (histogram of active vs empty non-null weight magnitudes)
# outputs (singular)
#	Negative Pearson Correlation weight differences.pdf; Positive Pearson Correlation weight differences.pdf (histograms of non-null active and empty FC weights)
#	Pearson Correlation weight magnitude differences.pdf (histograms of non-null active and empty FC weight magnitudes)
#	Null Average Magnitude Distributions all mice.png (histogram of active and empty FC weights in the null model)
#	Null Average Correlation Magnitude Difference Distribution all mice.png' (histogram of mean difference between active and empty FC magnitudes in null model)
#	Null vs Data Correlation Magnitude Differences.pdf (histogram of mean difference between active and empty FC magnitudes in null model compared to data)

# reorder functional networks by communities using the Brain Connectivity Toolbox (matlab -batch "FN_visualization")
# requires (for each mouse):
#	active_FN_pearson_corr; empty_FN_pearson_corr
# outputs (for each mouse):
#	FN_sort_pearson (.mat file containing the sort indices to index onto both active and empty FNs)

# then plot functional networks (python visualize_FNs.py)
# requires (for each mouse):
#	success_FN_pearson_corr; dry_FN_pearson_corr
#	FN_sort_pearson
# outputs (for each mouse):
# 	{mouseID} Active Grasp FN pearson_corr community sorted.pdf (active grasp FN matrix)
# 	{mouseID} Empty Grasp FN pearson_corr community sorted.pdf (empty grasph FN matrix)
# 	{mouseID} Active vs Empty Grasp FN {FN_method} community sorted.pdf (matrix of active grasp FN-empty grasp FN)

# find decodable FCs and plot the proportion decodable by FC type
python edge_decoder_svm.py && python plot_edge_decoding.py
# requires (for each mouse):
#	CellReg (folder including registered cell indices and red cell labels) - needed for all pairs of days
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl' (pickle file containing dictionary with unique/shared classifications of FC indices)

# outputs (for each mouse):
#	svm_decoding_accuracy_by_edge.npy (numpy array containing the decoding accuracies by FC across all folds with the shape FoldsxFCs)
# outputs (singular):
# 	mouse-averaged proportion of informative edges by edge type.pdf (barplot of the proportion of FCs that are decodable by FC type)


# FIGURE 5 AND RELATED SUPPLEMENTS

# perform condition-specific encoding models, LOO, and permutation model variants
# then plot results
python encoding_ridge_regression.py && python plot_encoding_results.py 
# requires (for each mouse):
#	CellReg (folder including registered cell indices and red cell labels) - needed for all pairs of days
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl' (pickle file containing dictionary with unique/shared classifications of FC indices)
#	sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl (pickle file containing dictionary by trial category of non-null FC indices)

# outputs (for each mouse):
#	{category}_trial_.3cv_splits_10_folds_distr_strat.pkl (pickle containing nested list of cv train/test splits for trial-wise stratified shuffle split)
#	{category}_trial_2Fold_cv_splits_10_repeats_distr_strat.pkl  (pickle containing nested list of cv train/test splits for trial-wise stratified repeated KFold split)
#	{category}_timepoint_{splitter}_cv_splits_{cv_folds}_folds.pkl (pickle containing nested list of cv train/test splits for timepoint-wise stratified repeated KFold or stratified shuffle split)
#	{category}_kinematics_only_lag{lag}_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy (neuronsXfolds array of r2 values for kinematics only encoding models)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy (neuronsXfolds array of r2 values for kinematics-matched category coupling encoding models)
#	{category}_kinematics_lag{lag}_and_opposite_coupling_trial_{cv_folds}Fold_{n_repeats}_repeats_ridge_regression_r2_values_distr_strat_NORMALIZED.npy (neuronsXfolds array of r2 values for kinematics-matched category coupling encoding models)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_all_edges_NORMALIZED.npy (neuronsXfolds array of r2 for all FC (nulls included) coupling encoding model)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_NORMALIZED.npy (neuronsXfoldsXLOO_models array of r2 for LOO by unique/shared FC type encoding model)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_uniqueness_dim_control_NORMALIZED.npy (neuronsXfoldsXcount-matched_LOO_models array of r2 for count-matched LOO by unique/shared FC type encoding model)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_NORMALIZED.npy (neuronsXfoldsXLOO_models array of r2 for LOO by FC weight magnitude encoding model)
#	{category}_kinematics_lag{lag}_and_coupling_trial_{splitter}_split_ridge_regression_r2_values_LOO_by_edge_strength_by_proportion_of_edges_{removed_edge_proportions}_NORMALIZED.npy (neuronsXfoldsXproportion_removedXLOO_models array of r2 for count-matched LOO by FC weight magnitude encoding model)
#	{category}_kinematics_lag{lag}_and_no_weights_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy (neuronsXfolds array of r2 values for coupling-based encoding with all non-diagonal weights=1)
#	{category}_kinematics_lag{lag}_and_permuted_weights_and_edges_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy (neuronsXfoldsXpermutations array of r2 values for permuted weights and connectivity encoding model)
#	{category}_kinematics_lag{lag}_and_permuted_weights_only_coupling_trial_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy (neuronsXfoldsXpermutations array of r2 values for permuted weights, preserved connectivity encoding model)
#	{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values_NORMALIZED.npy (neuronsXfolds array of r2 values for timepoint split matched-condition encoding model)
#	{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_predrop_r2_values_NORMALIZED.npy (neuronsXfolds array of r2 values for encoding model tested on pre-drop timepoints)
#	{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_postdrop_r2_values_NORMALIZED.npy (neuronsXfolds array of r2 values for encoding model tested on post-drop timepoints)
#	within vs opposite {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split NORMALIZED.png (scatter plot of within- vs opposite-condition coupling model performance per mouse)
# outputs (singular):
#	kinematic vs kinematic-coupling model performance median across folds.pdf (violin plot of kinematic-only model vs kinematic-coupling (matched-condition) model performance)
#	weight magnitude edges delta r2s violin dim matched.pdf (violin plot of LOO by weight magnitude delta r^2)
#	within vs opposite category couplings {which_measure} r2 scatterplot trial {n_repeats}x{n_folds}Fold split NORMALIZED.png (scatter plot of within- vs opposite-condition coupling model performance)
#	within vs opposite category couplings {which_measure} r2 scatterplot colored by density trial {n_repeats}x{n_folds}Fold split NORMALIZED.pdf (scatter plot of within- vs opposite-condition coupling model performance, colored by scatterplot density, with line of best fit)
#	shared unique delta r2s scatterplot magnitude and FC type.pdf (scatter plot of LOO delta r2 vs mean magnitude removed, colored by FC type removed)
#	encoding permutation comparison.pdf (violin plot of full kinematic-coupling model performance vs permutations)

# do combined-condition encoding models, then plot results vs condition-specific models
python encoding_combined_active_empty.py && python plot_combined_ridge.py
# requires (per mouse):
#	CellReg (folder including registered cell indices and red cell labels) - needed for all pairs of days
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)
#	unique_shared_by_null_comparison_{low_weight}{FN_method}.pkl' (pickle file containing dictionary with unique/shared classifications of FC indices)
#	sig_edges_comp_to_null_weighted_PDF_10000_pearson_corr.pkl (pickle file containing dictionary by trial category of non-null FC indices)
#	{category}_trial_2Fold_cv_splits_10_repeats_distr_strat.pkl  (pickle containing nested list of cv train/test splits for trial-wise stratified repeated KFold split)

# outputs (for each mouse):
#	combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_r2_values_normalized_features.npy (neuronsXfolds array of r2 values for combined-condition encoding model)
#	combined_active_empty_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_betas_normalized_features.npy (neuronsXfeaturesXfolds array of betas for combined-condition encoding model)
#	condition-specific vs condition-combined ridge encoding {which_measure} cv performance normalized.png (scatterplot of condition-specific vs condition-combined encoding performance)
#	condition-combined ridge encoding betas normalized.png (barplot of betas by feature for condition-combined encoding model)
#	condition-specific vs condition-combined ridge encoding {which_measure} cv performance violin normalized.png (violin plot of condition-specific vs condition-combined encoding performance)
# outputs (singular):
#	condition-specific vs condition-combined ridge encoding {which_measure} cv performance normalized.png (scatterplot of condition-specific vs condition-combined encoding performance)
#	condition-specific vs condition-combined ridge encoding {which_measure} cv performance colored by density normalized.pdf  (scatterplot of condition-specific vs condition-combined encoding performance, colored by scatterplot density, with line of best fit)
#	condition-specific vs condition-combined ridge {which_measure} cv encoding performance violin normalized.png (violin plot of condition-specific vs condition-combined encoding performance)


# FIGURE 6 AND RELATED SUPPLEMENTS 

# calculate d' across days of learning
python learning_discriminability.py
# requires (for each mouse): 
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)

# outputs (for each mouse):
#	day_pair_d_prime_diff_rel_to_peak; day_pair_d_prime_diff_rel_to_crossover.npy (numpy array containing d' for all cells per pair of days relative to peak/crossover)
#	Active-Empty D Prime Over Paired Days rel to peak; Active-Empty D Prime Over Paired Days rel to crossover.png (plot of population-averaged d' over days)
# outputs (singular):
#	Day Pair Active-Empty d prime Rel to peak.pdf; Day Pair Active-Empty d prime Rel to crossover.pdf (pdf file plotting population-averaged d' over days - combined across mice)

# calculate trial-to-trial variability across days of learning
python learning_within_day_PETH_correlations.py
# requires (for each mouse):
#	CellReg (folder including registered cell indices and red cell labels) - needed for all pairs of days
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)

# outputs (for each mouse):
#	within_day_pair_per_pair_PETH_correlations_subsampled_bootstrapped.pkl (pickled nested list containing PETH correlations for both trial classes across all pairs of days for 
#		data subsampled to the smaller trial class per pair of days)
#	within_day_pair_per_pair_PETH_correlations_bootstrapped.pkl (pickled nested list containing PETH correlations for both trial classes across all pairs of days)
#	Paired Days PETH Correlations Over Time 4 pre 4 post.png (plot of PETH correlations over time, average+-SEM across cells)
#	Paired Days PETH Correlations Over Time 4 pre 4 post subsampled.png (plot of PETH correlations subsampled to smaller trial class over time, average+-SEM across cells)
# outputs (singular):
#	Paired Days PETH Correlations Over Time 4 pre 4 post.png (plot of PETH correlations over time, average+-SEM across all cells from all mice)
#	Paired Days PETH Correlations Over Time 4 pre 4 post subsampled.png (plot of PETH correlations subsampled to smaller trial class over time, average+-SEM across all cells from all mice)
#	Paired Days PETH Correlations Over Time Aligned to Empty Peak.png (plot of PETH correlations over time aligned to empty peak, average+-SEM across all cells from all mice)
#	Paired Days PETH Correlations Over Time subsampled Aligned to Empty Peak.png (plot of PETH correlations subsampled to smaller trial class over time aligned to empty peak, average+-SEM across all cells from all mice)
#	Paired Days PETH Correlations Over Time Aligned to Crossover.pdf (plot of PETH correlations over time aligned to active-empty crossover, average+-SEM across all cells from all mice)
# 	Paired Days PETH Correlations Over Time subsampled Aligned to Crossover.pdf(plot of PETH correlations subsampled to smaller trial class over time aligned to active-empty crossover, average+-SEM across all cells from all mice)

# calculate day-to-day variability across pairs of days of learning
python learning_cross_day_carry_correlations.py
# requires (for each mouse):
#	CellReg (folder including registered cell indices and red cell labels) - needed for all pairs of days
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_carry_times (npy file containing all carry times for a given recording session, indexes onto cascade_spks)
#	calcium_carry_labels (npy file containing carry labels for a given recording session, indexes onto calcium_carry_times)

# outputs (for each mouse):
#	day_pair_correlations_subsampled_4_pre_4_post.pkl (pickled nested list containing cross-day PETH correlations for each pair with trials downsampled to the smaller category)
#	Paired PETH Correlations Over Time subsampled 4 pre 4 post.png (plot of cross-day PETH correlations over time subsampled to the smaller trial category, mean+-SEM across cells)
# outputs (singular):
#	Paired PETH Correlations Over Time subsampled 4 pre 4 post.png (plot of cross-day PETH correlations over time subsampled to the smaller trial category, mean+-SEM across all cells from all mice)
#	Paired PETH Correlations Over Time subsampled Aligned to Crossover.png (plot of cross-day PETH correlations over time aligned to crossover subsampled to the smaller trial category, mean+-SEM across all cells from all mice)
#	Paired PETH Correlations Over Time subsampled Aligned to Empty Peak.png (plot of cross-day PETH correlations over time aligned to empty peak subsampled to the smaller trial category, mean+-SEM across all cells from all mice)


# DROPS (FIGURE S4)

# find drop-modulated cells
python drop_analysis.py
# requires (per mouse):
#	CellReg (folder including registered cell indices and red cell labels)
#	cascade_spks (npy file containing all neural activity for a given recording session)
#	calcium_drop_times (npy file containing all drop times for a given recording session, indexes onto cascade_spks)
#	NaN_containing_cells_bool (boolean numpy array of which cells contain NaNs during relevant trials)
#	SVM_combined_active_grasp_modulated_cell_indices.npy (array of decodable cell indices)
#	{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_predrop_r2_values.npy
#	{category}_kinematics_and_coupling_timepoint_{splitter}_split_ridge_regression_postdrop_r2_values.npy
# outputs (per mouse):
#	Cell {neuron} All Days Drop PETH.png (drop PETH per cell)
#	All Days Population PETH.png (population-averaged PETH)
#	All Days Drop PETH.png (tiled population activity)
# 	All Days Drop PETH non-normalized.png (tiled population activity, not normalized per cell)
#	{mouseID}_all_days_drop_bon_mod_cells.npy (Boolean array of drop-modulated cells via Bonferroni-corrected ANOVA)
#	{mouseID}_all_days_drop_BH_mod_cells.npy (Boolean array of drop-modulated cells via false discovery rate corrected ANOVA)
#	pre vs post active {which_measure} r2 scatterplot trial split test {test_size}.png (scatterplot of pre- vs post-drop performance for active grasp models)
#	pre vs post empty {which_measure} r2 scatterplot trial split test {test_size}.png (scatterplot of pre- vs post-drop performance for empty grasp models)
#	pre vs post compare couplings {which_measure} r2 scatterplot trial split test {test_size}.png (scatterplot of pre- vs post-drop performance, colored by model training trial type)
#	active vs empty compare couplings {which_measure} r2 scatterplot trial split test {test_size}.png (scatterplot of active vs empty trained model performance, colored by pre- or post-drop testing)
# outputs (singular):
#	pre vs post empty couplings {which_measure} r2 scatterplot trial split test {test_size}.png (scatterplot of pre- vs post-drop performance for empty grasp models)
#	pre vs post active couplings {which_measure} r2 scatterplot trial split test {test_size}.png(scatterplot of pre- vs post-drop performance for active grasp models)
#	pre vs post compare couplings {which_measure} r2 scatterplot trial split test {test_size}.pdf (scatterplot of pre- vs post-drop performance, colored by model training trial type)
#	active vs empty compare pre-post {which_measure} r2 scatterplot trial split test {test_size}.pdf (scatterplot of active vs empty trained model performance, colored by pre- or post-drop testing)

# compare drop-modulated cell identities to population distributions
python drop_logit.py
# requires (per mouse):
#	NaN_containing_cells_bool (boolean numpy array of which cells contain NaNs during relevant trials)
#	SVM_combined_active_grasp_modulated_cell_indices.npy (array of decodable cell indices)
#	
# outputs (singular):
#	{cell_class} resampled vs drop cell counts.png