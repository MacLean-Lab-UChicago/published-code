'''
code from jidt package for calculating joint mutual information
'''
# mutual info
from jpype import *
import numpy as np
import sys
import src.IO
import math
import multiprocessing
import functools
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
data_dir = Path.cwd() / 'data'

def NatsToBits(x):
    ''' convert from nats to bits '''
    return x*(math.log2(math.e))

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
t_pre=4 # frames (30 fps)
t_post=4 # frames (30 fps)
time_in_trial=t_pre+t_post
# k=4 by default
n_repeats=100

compute_node_MI=True
# below is beta, tentatively does not work
compute_edge_jMI=False
compute_edge_condMI=False
find_synergy_from_jMI=False
find_synergy_from_cond_MI=False

if (not isJVMStarted()):
    # Add JIDT jar library to the path
    jarLocation = "/home/elizawiener/Documents/jidt/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation, convertStrings=True)

for mouseID in mice:
    # load in the data
    days = src.IO.get_carry_days(mouseID)
    drive = data_dir + '/neural'
    mouse_dir = drive + '/' + mouseID + '/'
    save_dir = mouse_dir + 'carry_analysis/MI/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    reg_inds = src.utils.load_registered_cells(mouse_dir, days)
    neural_activity_by_trial = [[] for day in days]
    category_labels = [[] for day in days]
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
        carry_times = carry_times[carry_labels!=0]
        carry_labels = carry_labels[carry_labels!=0]-1
        category_labels_by_time = np.zeros((len(carry_times)*time_in_trial))

        # get trial-specific neural data and behavioral data
        temp_spks = [[] for time in carry_times]
        for carry_i, carry in enumerate(carry_times.astype(int)):
            temp_spks[carry_i] = reg_spks[:, (carry-t_pre):(carry+t_post)]
            category_labels_by_time[(carry_i*time_in_trial):((carry_i+1)*time_in_trial)] = carry_labels[carry_i]
        neural_activity_by_trial[i] = np.hstack(temp_spks)
        category_labels[i] = np.hstack(category_labels_by_time)

    neural_activity = np.hstack(neural_activity_by_trial)
    category_labels = np.hstack(category_labels)

    # remove neurons with only NaN for activity
    activity_nonan = neural_activity[~np.isnan(neural_activity).any(axis=1), :]
    print(np.sum(np.isnan(neural_activity).any(axis=1)), ' NaN-containing cells removed')
    NaN_indices = np.where(np.isnan(neural_activity).any(axis=1))[0]
    # print(NaN_indices)
    n_neurons = activity_nonan.shape[0]
    print('n cells: ', n_neurons)

    # 0. Load/prepare the data:
    # format to pairs and trial labels by time
    destination = JArray(JInt, 1)(np.array(category_labels, dtype=int).tolist())
    destination_continuous = JArray(JDouble, 1)(category_labels.tolist())

    if compute_node_MI:
        # compute MI b/w neuron and trial category for all neurons
        MI = np.zeros(n_neurons)*np.nan
        MI_std = np.zeros(n_neurons)*np.nan
        MI_continuous = np.zeros(n_neurons)*np.nan
        MI_continuous_std = np.zeros(n_neurons)*np.nan

        for neuron in range(n_neurons):
            result = np.zeros(n_repeats)*np.nan
            source = JArray(JDouble, 1)(activity_nonan[neuron, :].tolist())

            # 1. Construct the calculator:
            calcClass = JPackage("infodynamics.measures.mixed.kraskov").MutualInfoCalculatorMultiVariateWithDiscreteKraskov
            calc = calcClass()
            # 2. Set any properties to non-default values:
            # No properties were set to non-default values
            for i in range(n_repeats):
                # 3. Initialise the calculator for (re-)use:
                calc.initialise(1, 2) # 1 dimensional data, 2 discrete base states
                # 4. Supply the sample data:
                calc.setObservations(source, destination)
                # 5. Compute the estimate:
                result[i] = calc.computeAverageLocalOfObservations()
            result = NatsToBits(result)
            MI[neuron] = np.mean(result)
            # print(neuron, MI[neuron])
            MI_std[neuron] = np.std(result)

            # 1. Construct the calculator:
            calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
            calc = calcClass()
            # 2. Set any properties to non-default values:
            # No properties were set to non-default values
            for i in range(n_repeats):
                # 3. Initialise the calculator for (re-)use:
                calc.initialise() # 1 dimensional data, 2 discrete base states
                # 4. Supply the sample data:
                calc.setObservations(source, destination_continuous)
                # 5. Compute the estimate:
                result[i] = calc.computeAverageLocalOfObservations()
            result = NatsToBits(result)
            # print('assuming continuous: ', np.mean(result))
            MI_continuous[neuron] = np.mean(result)
            MI_continuous_std[neuron] = np.std(result)

        print('median MI b/w neurons and trial cat: ', np.median(MI), ' bits')
        np.save(save_dir + 'node_trialcat_MI.npy', MI)
        np.save(save_dir + 'node_trialcat_MI_std.npy', MI_std)
        np.save(save_dir + 'node_trialcat_MI_continous_assumption.npy', MI_continuous)
        np.save(save_dir + 'node_trialcat_MI_std_continous_assumption.npy', MI_continuous_std)

    # BETA, do not trust the outputs
    if compute_edge_jMI:
        start_joint = time.time()
        # compute conditional MI b/w joint of two neurons and trial category for all neurons
        joint_MI = np.zeros((n_neurons, n_neurons))*np.nan
        joint_MI_std = np.zeros((n_neurons, n_neurons))*np.nan
        joint_MI_continuous = np.zeros((n_neurons, n_neurons))*np.nan
        joint_MI_continuous_std = np.zeros((n_neurons, n_neurons))*np.nan
        # print(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1])
        # test = np.zeros((n_neurons, n_neurons))
        # test[np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1]] = 1
        # plt.imshow(test)
        # plt.show()

        for edge, (i, j) in enumerate(zip(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1])):
            result=np.zeros(n_repeats)*np.nan
            if (edge+1)%1000==0:
                print(f'working on edge {edge+1}/{len(np.triu_indices(n_neurons, k=1)[0])}')
                print(f'{time.time()-start_joint} s spent calculating joint MI')
            #print('neurons ', i, j)
            source = JArray(JDouble, 2)(activity_nonan[[i, j], :].T.tolist())

            # 1. Construct the calculator:
            calcClass = JPackage("infodynamics.measures.mixed.kraskov").MutualInfoCalculatorMultiVariateWithDiscreteKraskov
            calc = calcClass()
            # 2. Set any properties to non-default values:
            # no non-default (keeping k at 4)
            for repeat in range(n_repeats):
                # 3. Initialise the calculator for (re-)use:
                calc.initialise(2, 2) # 2d source, 2 discrete states
                # 4. Supply the sample data:
                calc.setObservations(source, destination)
                # 5. Compute the estimate:
                result[repeat] = calc.computeAverageLocalOfObservations()
            if np.isnan(result).any():
                print(result)
            result1 = NatsToBits(result)
            jMI = np.mean(result1)
            if np.isnan(jMI):
                print('jMI NaN! neurons: ', i, j, result1)
            joint_MI[i, j] = jMI
            joint_MI[j, i] = jMI
            joint_MI_std[i, j] = np.std(result1)
            joint_MI_std[j, i] = np.std(result1)

            result=np.zeros(n_repeats)*np.nan
            # 1. Construct the calculator:        
            calcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
            calc = calcClass()
            # 2. Set any properties to non-default values:
            # no non-default (keeping k at 4)
            for repeat in range(n_repeats):
                # 3. Initialise the calculator for (re-)use:
                calc.initialise(2, 1) # 2d source, 1d destination (?)
                # 4. Supply the sample data:
                calc.setObservations(source, destination_continuous)
                # 5. Compute the estimate:
                result[repeat] = calc.computeAverageLocalOfObservations()
            result2 = NatsToBits(result)
            jMI_continous = np.mean(result2)
            joint_MI_continuous[i, j] = jMI_continous
            joint_MI_continuous[j, i] = jMI_continous
            joint_MI_continuous_std[i, j] = np.std(result2)
            joint_MI_continuous_std[j, i] = np.std(result2)

        print('median joint MI b/w neurons and trial cat: ', np.nanmedian(joint_MI), ' bits')
        # plt.imshow(joint_MI)

        np.save(save_dir + 'edge_trialcat_jMI.npy', joint_MI)
        np.save(save_dir + 'edge_trialcat_jMI_std.npy', joint_MI_std)
        np.save(save_dir + 'edge_trialcat_jMI_continous_assumption.npy', joint_MI_continuous)
        np.save(save_dir + 'edge_trialcat_jMI_std_continous_assumption.npy', joint_MI_continuous_std)

    # BETA, do not trust the outputs
    if compute_edge_condMI:
        start_conditional = time.time()
        # compute conditional MI b/w joint of two neurons and trial category for all neurons
        cond_MI_continuous = np.zeros((n_neurons, n_neurons))*np.nan
        cond_MI_continuous_std = np.zeros((n_neurons, n_neurons))*np.nan

        for edge, (i, j) in enumerate(zip(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1])):
            result=np.zeros(n_repeats)*np.nan
            if (edge+1)%1000==0:
                print(f'working on edge {edge+1}/{len(np.triu_indices(n_neurons, k=1)[0])}')
                print(f'{time.time()-start_conditional} s spent calculating joint MI')
            #print('neurons ', i, j)
            source = JArray(JDouble, 1)(activity_nonan[i, :].T.tolist())
            conditional = JArray(JDouble, 1)(activity_nonan[j, :].T.tolist())

            # 1. Construct the calculator:        
            calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
            calc = calcClass()
            # 2. Set any properties to non-default values:
            # no non-default (keeping k at 4)
            for repeat in range(n_repeats):
                # 3. Initialise the calculator for (re-)use:
                calc.initialise() # 2d source, 1d destination (?)
                # 4. Supply the sample data:
                calc.setObservations(source, destination_continuous, conditional)
                # 5. Compute the estimate:
                result[repeat] = calc.computeAverageLocalOfObservations()
            result2 = NatsToBits(result)
            cond_MI = np.mean(result2)
            cond_MI_continuous[i, j] = cond_MI
            cond_MI_continuous_std[i, j] = np.std(result2)

        print('median joint MI b/w neurons and trial cat: ', np.nanmedian(cond_MI_continuous), ' bits')
        # plt.imshow(joint_MI)

        np.save(save_dir + 'edge_trialcat_conditional_MI_continous_assumption.npy', cond_MI_continuous)
        np.save(save_dir + 'edge_trialcat_conditional_MI_std_continous_assumption.npy', cond_MI_continuous_std)

    # BETA, do not trust the outputs
    if find_synergy_from_cond_MI:
        MI = np.load(save_dir + 'node_trialcat_MI_continous_assumption.npy')
        print(np.median(MI))
        cMI = np.load(save_dir + 'edge_trialcat_conditional_MI_continous_assumption.npy')
        print(np.median(cMI[np.triu_indices(n_neurons, k=1)]))
        MI_std = np.load(save_dir + 'node_trialcat_MI_std_continous_assumption.npy')
        cMI_std = np.load(save_dir + 'edge_trialcat_conditional_MI_std_continous_assumption.npy')
        # find syngergistic edges
        synergistic_edges = np.zeros((n_neurons, n_neurons), dtype=bool)*np.nan
        for i, j in zip(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1]):
            if cMI[i, j]>MI[i]:
                synergistic_edges[i, j] = True
                synergistic_edges[j, i] = True
            else:
                synergistic_edges[i, j] = False
                synergistic_edges[j, i] = False
        np.save(save_dir + 'synergistic_edge_bool_from_continuous_conditional.npy', synergistic_edges)
        print(f'{np.sum(synergistic_edges[np.triu_indices(n_neurons, k=1)])}/{len(synergistic_edges[np.triu_indices(n_neurons, k=1)])} edges are synergistic')

        # find syngergistic edges
        sig_synergistic_edges = np.zeros((n_neurons, n_neurons), dtype=bool)*np.nan
        for i, j in zip(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1]):
            if (cMI[i, j]-cMI_std[i, j])>(MI[i]+MI_std[i]):
                sig_synergistic_edges[i, j] = True
                sig_synergistic_edges[j, i] = True
            else:
                sig_synergistic_edges[i, j] = False
                sig_synergistic_edges[j, i] = False
        print(f'{np.sum(sig_synergistic_edges[np.triu_indices(n_neurons, k=1)])}/{len(sig_synergistic_edges[np.triu_indices(n_neurons, k=1)])} edges are synergistic')


    # BETA, do not trust the outputs
    if find_synergy_from_jMI:
        MI = np.load(save_dir + 'node_trialcat_MI.npy')
        print(np.median(MI))
        joint_MI = np.load(save_dir + 'edge_trialcat_jMI.npy')
        print(np.median(joint_MI))
        # find syngergistic edges
        synergistic_edges = np.zeros((n_neurons, n_neurons), dtype=bool)*np.nan
        for i, j in zip(np.triu_indices(n_neurons, k=1)[0], np.triu_indices(n_neurons, k=1)[1]):
            if joint_MI[i, j]>(MI[i]+MI[j]):
                synergistic_edges[i, j] = True
                synergistic_edges[j, i] = True
            else:
                synergistic_edges[i, j] = False
                synergistic_edges[j, i] = False
        np.save(save_dir + 'synergistic_edge_bool.npy', synergistic_edges)
        print(f'{np.sum(synergistic_edges[np.triu_indices(n_neurons, k=1)])}/{len(synergistic_edges[np.triu_indices(n_neurons, k=1)])} edges are synergistic')
