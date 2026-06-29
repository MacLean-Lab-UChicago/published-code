import numpy as np
import matplotlib.pyplot as plt
import src.utils
import os
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
carry_classes = [1, 2] # drop is 0, active is 1, empty is 2

normalize=True

t_pre = 4
t_post = 4

for mouseID in mice:
    mouse_dir = data_dir + '/neural/' + mouseID + '/'
    days = src.IO.get_carry_days(mouseID)

    save_dir = data_dir + '/results/' + mouseID + '/'
    if not os.path.isdir(save_dir + 'figures/'):
        os.mkdir(save_dir + 'figures/')
    reg_inds = src.utils.load_registered_cells(mouse_dir, days)

    F_multi = [[] for i in range(len(days))]
    carry_time_multi = [[[] for i in range(len(days))] for i in range(len(carry_classes))]
    carry_labels_multi = [[] for i in range(len(days))]
    for i, day in enumerate(days):
        s2p_fld = src.IO.get_s2p_fld(mouseID, day)
        # load in data per day
        f = np.load(s2p_fld + 'cascade_spks.npy')
        F_reg = f[reg_inds[i], :]
        F = F_reg
        carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
        carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
        # format per day
        F_multi[i] = F
        for c_class, cat in enumerate(carry_classes):
            carry_time_multi[c_class][i] = carry_times[carry_labels==cat]

    fig, ax = plt.subplots(2, 2, figsize=(6.25, 5.55))
    column = -1
    for c_class, cat in enumerate(carry_classes):
        column+=1
        row=-1

        if cat==0:
            title='drop'
        elif cat==1:
            title='active grasp'
        elif cat==2:
            title='empty grasp'

        for class_to_sort_by in carry_classes:
            if class_to_sort_by==0:
                c_class_name='drop'
            elif class_to_sort_by==1:
                c_class_name='active grasp'
            elif class_to_sort_by==2:
                c_class_name='empty grasp'

            row+=1
            if cat == class_to_sort_by:
                sort_ind = None 
            else:
                sort_ind = np.load(save_dir + f'carry_{class_to_sort_by}_sort_indices.npy')

            # calculate PETH and sort indices
            [F_sorted, indices] = src.utils.plot_PETH_multiday(days, F_multi, carry_time_multi[c_class], t_pre=t_pre, t_post=t_post, sort_ind=sort_ind, norm=normalize)
            
            # save out sort indices
            if cat == class_to_sort_by:
                np.save(save_dir + f'carry_{class_to_sort_by}_sort_indices.npy', indices, allow_pickle=False)

            # plot tiled PETH
            im = ax[row, column].imshow(F_sorted[~np.isnan(F_sorted).all(axis=1), :], aspect='auto')
            ax[row, column].vlines(0+t_pre, 0, len(F_sorted[~np.isnan(F_sorted).all(axis=1), :])-1, color='white', linestyles='dashed')
            locations = [0+1, 0+t_pre, t_pre+t_post-1]
            labels = [-(t_pre-1)*1000/30, 0, (t_post-1)*1000/30]
            
            if column==0:
                ax[row, column].set_ylabel(f'Neurons sorted\nby {c_class_name}', fontsize=12)
            else:
                ax[row, column].set_yticks([])
            if row==2 and column==0:
                ax[row, column].set_xticks(locations, labels=labels, fontsize=10)
                ax[row, column].set_xlabel("Time (ms)", fontsize=12)
            else:
                ax[row, column].set_xticks([])
            if row==0:
                ax[row, column].set_title(title)

    #fig.text(0.5, 0, 'Time From Reach Onset (ms)', ha='center', fontsize=12)
    #fig.text(0, 0.5, 'common Y', va='center', rotation='vertical')
    fig.colorbar(im, ax=ax)
    #plt.tight_layout()
    if normalize:
        plt.savefig(save_dir + '/figures/active_empty_carry_PETHs_norm.png', dpi=660)
    else:
        plt.savefig(save_dir + '/figures/active_empty_carry_PETHs.png')
    plt.show()