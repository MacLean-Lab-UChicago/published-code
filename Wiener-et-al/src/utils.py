import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, lfilter, freqz
import src.IO
import scipy
from scipy.interpolate import PchipInterpolator
import glob


def get_reg_ind(reg_file):
    '''
    loads cell reg output from the cellRegistered .mat file
     and converts it to a list of arrays of indices. The list has length
     ndays included, and each array indexes the registered cells for that day,
     all in the same order
    ''' 
    import mat73
    reg = mat73.loadmat(reg_file)
    reg = reg['cell_registered_struct']['cell_to_index_map']
    reg_list = []
    for i in range(reg.shape[1]):
        reg_list.append(reg[:,i]>0)
    cells_in_all = np.all(np.array(reg_list),axis=0)
    print('no. registered cells: ',np.sum(cells_in_all))
    ind_list = []
    for i in range(reg.shape[1]):
        ind_list.append(reg[cells_in_all,i].astype(int)-1)#minus one because matlab uses 1-indexing
    return ind_list 

def load_registered_and_red_cells(mouse_directory, days):
    if len(days)>2:
        reg_paths = sorted(glob.glob(mouse_directory+'CellRegResults/*/cellRegistered*.mat'))
        path = [path for path in reg_paths if '_to_'.join([days[0],days[-1]]) in path]
    else:
        reg_paths = sorted(glob.glob(mouse_directory+'CellRegResults/*/cellRegistered*.mat'))
        path = [path for path in reg_paths if '_'.join([days[0],days[-1]]) in path]
    if len(path)==0:
        print(reg_paths)
        if len(days)>2:
            print('_to_'.join([days[0],days[-1]]))
        else:
            print('_'.join([days[0],days[-1]]))
        raise Warning('no paths')
    elif len(path)>1:
        print(path)
        path = path[-1] #most recent
    else:
        path = path[0]
    reg_inds = get_reg_ind(path)

    # load in red cell labels 
    red_cells = load_red_cells(mouse_directory, days)
    return reg_inds, red_cells

def load_registered_cells(mouse_directory, days):
    if len(days)>2:
        reg_paths = sorted(glob.glob(mouse_directory+'CellRegResults/*/cellRegistered*.mat'))
        path = [path for path in reg_paths if '_to_'.join([days[0],days[-1]]) in path]
    else:
        reg_paths = sorted(glob.glob(mouse_directory+'CellRegResults/*/cellRegistered*.mat'))
        path = [path for path in reg_paths if '_'.join([days[0],days[-1]]) in path]
    if len(path)==0:
        print(reg_paths)
        if len(days)>2:
            print('_to_'.join([days[0],days[-1]]))
        else:
            print('_'.join([days[0],days[-1]]))
        raise Warning('no paths')
    elif len(path)>1:
        print(path)
        path = path[-1] #most recent
    else:
        path = path[0]
    reg_inds = get_reg_ind(path)
    return reg_inds

def load_red_cells(mouse_directory, days):
    if len(days)>2:    
        red_cells = np.load(f'{mouse_directory}CellRegResults/{days[0]}_to_{days[-1]}/red_cell_labels_on_reg_ind.npy')
    else:
        red_cells = np.load(f'{mouse_directory}CellRegResults/{days[0]}_{days[-1]}/red_cell_labels_on_reg_ind.npy')
    return red_cells

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_PETH(F,reach_starts,t_pre=15,t_post=30,sort=True,sort_ind=None,norm=False):
    sum_F = np.zeros([F.shape[0],t_pre+t_post])
    for t in reach_starts.astype(int):
        if (t+t_post < (F.shape[1]-32)) and (t-t_pre > 32):
            sum_F = sum_F + F[:,(t-t_pre):(t+t_post)]
    avg_F = sum_F/len(reach_starts)
    #normalize each cell's avg activity
    if norm:
        for neuron in range(avg_F.shape[0]):
            avg_F[neuron] = normalize(avg_F[neuron])
    #now sort
    if sort:
        if sort_ind is not None:
            #sort by given sorting indices
            avg_F_sorted = avg_F[sort_ind,:]
            return avg_F_sorted,sort_ind
        else:
            #sort by latency
            ind_maxF = np.argmax(avg_F,axis=1)
            avg_F_sorted = avg_F[np.argsort(ind_maxF),:]
            return avg_F_sorted, np.argsort(ind_maxF)
    else:
        return avg_F, None

def plot_PETH_multiday(days, F_multi,reach_starts_multi,t_pre=15,t_post=30,sort=True,sort_ind=None,norm=False):
    '''
    requires F_multi to be set up as a nested list [[F_day1], [F_day2], ...]
    and reach_starts_multi similarly as [[reach_starts_day1], [r_s_day2], ...]
    '''
    sum_F = np.zeros([F_multi[0].shape[0],t_pre+t_post])
    reach_count = 0
    for day_i in range(len(days)):
        for t in reach_starts_multi[day_i].astype(int):
            if (t+t_post < (F_multi[day_i].shape[1]-32)) and (t-t_pre > 32):
                sum_F = sum_F + F_multi[day_i][:,(t-t_pre):(t+t_post)]
                reach_count+=1
    # print('n reaches: ', reach_count)
    avg_F = sum_F/reach_count
    #normalize each cell's avg activity
    if norm:
        for neuron in range(avg_F.shape[0]):
            avg_F[neuron] = normalize(avg_F[neuron])
    #now sort
    if sort:
        if sort_ind is not None:
            #sort by given sorting indices
            avg_F_sorted = avg_F[sort_ind,:]
            return avg_F_sorted,sort_ind
        else:
            #sort by latency
            ind_maxF = np.argmax(avg_F,axis=1)
            avg_F_sorted = avg_F[np.argsort(ind_maxF),:]
            return avg_F_sorted, np.argsort(ind_maxF)
    else:
        return avg_F, None

def get_labels_for_behavior(event_labels,beh):
    '''boolean inds for different behaviors to index event times'''
    if beh=='reach':
        labels = event_labels==0 
    elif beh == 'all_reach':
        #returns ind for all reaching movements (reach, grasp, and carry)
        labels = np.logical_or(np.logical_or(event_labels==0,event_labels==1),event_labels==2)
    elif beh=='grasp':
        #returns ind for grasps
        labels = event_labels==1
    elif beh=='carry':
        #returns ind for carries
        labels = event_labels==2
    elif beh=='eating':
        #returns ind for single brief instances of food manipulation 
        #(e.g a single turn of the pellet)
        labels = event_labels==5                    
    elif beh=='grooming':
        #returns ind for grooming
        labels = event_labels==6
    elif beh=='fidget':
        #returns ind for fidgets
        labels = event_labels==4 
    elif beh == 'non_movement':
        #returns ind for non_movement - this is not comprehensive
        labels = event_labels==3
    elif beh == 'misclass':
        #ind that were misclassified by the random forest behavior class
        #model, and caught in the manual s/f classification
        labels = event_labels==7
    elif beh == 'reach_grasp':
        #ind for reach or grasp
        labels = np.logical_or(event_labels==0,event_labels==1)
    elif beh == 'grasp_carry':
        #ind for grasp or carry
        labels = np.logical_or(event_labels==1,event_labels==2)
    elif beh == 'non_reach':
        #ind for eating, grooming, fidget, or nonmovement
        #(e.g. for reach/nonreach anova)
        labels = np.logical_or(np.logical_or(event_labels==3,event_labels==4),
            np.logical_or(event_labels==5,event_labels==6))
    return labels 

def get_fr_for_behavior(spks, reach_starts,t_pre,t_post):
    #returns avg spks with shape n_cells x n_timepoints
    sum_spks = np.zeros([spks.shape[0],t_pre+t_post])
    for t in reach_starts.astype(int):
        if (t+t_post < (spks.shape[1]-32)) and (t-t_pre > 32):
            sum_spks = sum_spks + spks[:,(t-t_pre):(t+t_post)]
    avg_spks = sum_spks/len(reach_starts)
    return avg_spks

def get_trial_fr_for_behavior(spks, reach_starts,t_pre,t_post):
    #returns avg spks with shape n_cells x n_trials
    avg_spks = np.zeros([spks.shape[0],reach_starts.shape[0]])
    tr_time = (t_pre+t_post)/30
    for i,t in enumerate(reach_starts.astype(int)):
        if (t+t_post < (spks.shape[1]-32)) and (t-t_pre > 32):
            avg_spks[:,i] = np.nansum(spks[:,(t-t_pre):(t+t_post)],axis=1)/tr_time
    return avg_spks

def get_cat_spks_for_behavior(spks,reach_starts,t_pre,t_post):
    reach_starts = reach_starts[(reach_starts+t_post)<(spks.shape[1]-32)]
    reach_starts = reach_starts[(reach_starts-t_pre)>32]
    duration = t_pre+t_post
    spks_cat = np.zeros([spks.shape[0],reach_starts.shape[0]*duration])
    ind = np.arange(0,spks_cat.shape[1]+1,duration).astype(int)
    for i,t in enumerate(reach_starts.astype(int)):
        spks_cat[:,ind[i]:ind[i+1]] = spks[:,(t-t_pre):(t+t_post)]
    return spks_cat

def anova(reach_trials,other_trials,n_bootstraps = 100,subsample=False,log=False):

    #plot_residuals_and_std(reach_trials,other_trials)
    if log:
        reach_trials = np.log10(reach_trials+1e-6)
        other_trials = np.log10(other_trials+1e-6)
    
    if subsample:
        p = np.zeros((reach_trials.shape[0],n_bootstraps))
        for n in range(n_bootstraps):
            if reach_trials.shape[1]<other_trials.shape[1]:
                subsample_ind = np.random.choice(other_trials.shape[1],size=reach_trials.shape[1],replace=False)
                other_trials = other_trials[:,subsample_ind]
            elif reach_trials.shape[1]>other_trials.shape[1]:
                subsample_ind = np.random.choice(reach_trials.shape[1],size=other_trials.shape[1],replace=False)
                reach_trials = reach_trials[:,subsample_ind]
            F = stats.f_oneway(reach_trials,other_trials,axis=1,nan_policy='omit') #this should give me a stat for each cell
            p[:,n] = F.pvalue
        mean_p = np.mean(p,axis=1)
        return mean_p
    else:
        F = stats.f_oneway(reach_trials,other_trials,axis=1,nan_policy='omit') #this should give me a stat for each cell
        return F.pvalue

def get_flat_upper_tri(matrix):
    return matrix[np.triu_indices(matrix.shape[0],k=1)].flatten()

def binarize_spikes(full_Cascade, Cascade, percent_threshold):
    thresh = np.nanpercentile(full_Cascade, percent_threshold)
    #print(thresh)
    raster = Cascade.copy()
    raster[Cascade<thresh]=0
    raster[Cascade>=thresh]=1
    return raster

def calculate_MI(spikes_x, spikes_y):
    # get individual probabilities
    p_x1 = np.mean(spikes_x)
    p_y1 = np.mean(spikes_y)
    p_x0 = 1-p_x1
    p_y0 = 1-p_y1

    # get joint probabilities
    p_1_1 = np.mean(np.logical_and(spikes_x, spikes_y))
    p_0_0 = np.mean(np.logical_and(np.logical_not(spikes_x), np.logical_not(spikes_y)))
    p_0_1 = np.mean(np.logical_and(np.logical_not(spikes_x), spikes_y))
    p_1_0 = np.mean(np.logical_and(spikes_x, np.logical_not(spikes_y)))

    MI = 0
    if p_1_1>0:
        MI+=p_1_1*np.log2(p_1_1/(p_x1*p_y1))
    if p_0_0>0:
        MI+=p_0_0*np.log2(p_0_0/(p_x0*p_y0))
    if p_0_1>0:
        MI+=p_0_1*np.log2(p_0_1/(p_x0*p_y1))
    if p_1_0>0:
        MI+=p_1_0*np.log2(p_1_0/(p_x1*p_y0))

    return MI

def get_informative_cells_no_NaNs(s2p_fld):
    inform_neurons = np.load(s2p_fld + 'SVM_combined_active_grasp_modulated_cell_indices.npy')
    NaN_cells = np.where(np.load(s2p_fld + 'NaN_containing_cells_bool.npy'))[0]
    if len(NaN_cells)>0:
        for NaN_cell in NaN_cells:
            inform_neurons[inform_neurons>NaN_cell]-=1
    return inform_neurons

def adjust_indices_for_NaN_cell_removal(indices, NaN_cell_indices):
    for NaN_cell in sorted(NaN_cell_indices, reverse=True):
        indices[indices>NaN_cell]-=1
    return indices

def adjust_indices_for_NaN_cell_removal_with_mouseID(indices, mouseID):
    day = src.IO.get_carry_days(mouseID)[-1]
    s2p_fld = src.IO.get_s2p_fld(mouseID, day)
    NaN_cell_indices = np.where(np.load(s2p_fld + 'NaN_containing_cells_bool.npy'))[0]
    for NaN_cell in sorted(NaN_cell_indices, reverse=True):
        indices[indices>NaN_cell]-=1
    return indices

def find_long_nan_sequences(arr, min_length=3):
    ''' from ChatGPT, prompt: find the start and end indexes (inclusive) of a sequence of NaNs 
    in a np array only if that sequence of NaNs consists of more than min_length in a row '''
    isnan = np.isnan(arr)
    diff = np.diff(np.concatenate(([0], isnan.view(np.int8), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]-1

    # Keep only sequences that are long enough
    long_mask = (ends - starts) > min_length
    return list(zip(starts[long_mask], ends[long_mask]))

def split_data(features, day_labels, n_chunks_per_day=10, split_ratio=0.7):
    '''generates train and test indices (approximately) evenly distributed across days'''
    '''stopped actually using it, used StratifiedSplit from scipy instead'''
    if not (split_ratio*100)%n_chunks_per_day==0:
        raise Warning(f'Split Ratio of {split_ratio} does not work with {n_chunks_per_day} chunks')

    time_points = np.arange(features.shape[0])

    #print(features[:,-1])
    #print(np.sum(features[:, -1]==2)/features.shape[0])

    chunked_day_indices = [[[] for chunk in range(n_chunks_per_day)] for day in np.unique(day_labels)]
    times_per_day = np.zeros(len(np.unique(day_labels)))
    for day_i, day in enumerate(np.unique(day_labels)):
        time_for_day = time_points[day_labels==day]
        #print(time_for_day)
        n_timepoints = np.sum(day_labels==day)
        chunk_size = n_timepoints//n_chunks_per_day
        remainder = n_timepoints%n_chunks_per_day
        for chunk in range(n_chunks_per_day):
            chunked_day_indices[day_i][chunk] = time_for_day[(chunk*chunk_size):((chunk+1)*chunk_size)]
        if remainder>0:
            for leftover in range(remainder):
                chunked_day_indices[day_i][leftover] = np.append(chunked_day_indices[day_i][leftover], time_for_day[(chunk+1)*chunk_size+leftover])

    #print(chunked_day_indices[0][0])

    # randomly group different chunks from each day
    chunked_indices = [[] for chunk in range(n_chunks_per_day)]
    for day_i in range(len(np.unique(day_labels))):
        rand_order = np.arange(n_chunks_per_day)
        np.random.shuffle(rand_order)
        #print(rand_order)
        #print(chunked_indices)
        for chunk_i in range(n_chunks_per_day):
            chunked_indices[chunk_i].extend(chunked_day_indices[day_i][rand_order[chunk_i]])

    train_chunks = rand_order[:int(n_chunks_per_day*split_ratio)]
    test_chunks = [i for i in np.arange(n_chunks_per_day) if i not in train_chunks]

    # print(train_chunks)
    # print(test_chunks)

    train_indices = np.concatenate([chunked_indices[train_chunk] for train_chunk in train_chunks])
    #print(train_indices)
    #print(len(train_indices))
    test_indices = np.concatenate([chunked_indices[test_chunk] for test_chunk in test_chunks])
    #print(test_indices)
    # print(len(test_indices))
    # print(f'train/test drops: {np.sum(fe
