'''
make a barplot (or maybe a pie chart?) from the results of the edge decoder
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import src.IO

mice = ('mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549')
categories = ['Decodeable', 'Non-Decodable']
edge_cat = ['Shared Edge, Decodable', 'Unique Edge, Decodable', 'Sign Flip, Decodable', 'Zero, Decodable', 'Shared Edge, Non-Decodable', 'Unique Edge, Non-Decodable', 'Sign Flip, Non-Decodable', 'Zero, Non-Decodable']
fig_save_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
edge_types = ['Overall', 'Non-Zero', 'Null', 'Unique', 'Shared', 'Sign Flip']

prop_informative = np.zeros((len(edge_types), len(mice)))
count_informative = np.zeros((len(edge_types), len(mice)))
edge_type_counts = np.zeros((len(edge_types), len(mice)))
mod_counts = {key:np.zeros(len(mice)) for key in categories}
edge_cat_mod_counts = {key:np.zeros(len(mice)) for key in edge_cat}
total_edge_counts = np.zeros(len(mice))
for mouse_i, mouseID in enumerate(mice):
	days = src.IO.get_carry_days(mouseID)
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	edges = src.IO.load_pickle(mouse_dir + f'carry_analysis/unique_shared_by_null_comparison_weighted_PDF_10000_pearson_corr.pkl')
	calcium_data_path = mouse_dir + days[-1]
	if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
		s2p_fld = calcium_data_path + '/'
	else:
		s2p_fld = calcium_data_path + '/recording/tifs/suite2p/plane0/'
	accuracy = np.load(s2p_fld + 'svm_decoding_accuracy_by_edge.npy')
	accuracy_by_edge = np.mean(accuracy, axis=0)
	decodable = (accuracy_by_edge-(stats.sem(accuracy, axis=0)*1.96))>0.5 # accuracy-95% CI > chance
	mod_counts['Decodeable'][mouse_i] = np.sum(decodable)
	mod_counts['Non-Decodable'][mouse_i] = np.sum(~decodable) 
	total_edge_counts[mouse_i] = len(decodable)

	print(np.sum(decodable)/len(decodable))
	prop_informative[0, mouse_i] = np.sum(decodable)/len(decodable)
	count_informative[0, mouse_i] = np.sum(decodable)
	edge_type_counts[0, mouse_i] = len(decodable)

	# index shared and unique edges
	{'unique_wet':[], 'unique_dry':[], 'shared':[], 'sign_flip':[], 'both_zero':[]}
	unique_edge_bool = np.zeros(len(decodable),dtype=bool)
	unique_edge_indices = [i for i in range(len(unique_edge_bool)) if i in edges['unique_wet'] or i in edges['unique_dry']]
	unique_edge_bool[unique_edge_indices] = True
	
	shared_edge_bool = np.zeros(len(decodable),dtype=bool)
	shared_edge_indices = [i for i in range(len(shared_edge_bool)) if i in edges['shared']]
	shared_edge_bool[shared_edge_indices] = True
	
	sf_edge_bool = np.zeros(len(decodable),dtype=bool)
	sf_edge_indices = [i for i in range(len(unique_edge_bool)) if i in edges['sign_flip']]
	sf_edge_bool[sf_edge_indices] = True
	
	z_edge_bool = np.zeros(len(decodable),dtype=bool)
	z_edge_indices = [i for i in range(len(unique_edge_bool)) if i in edges['both_zero']]
	z_edge_bool[z_edge_indices] = True

	edge_cat_mod_counts['Shared Edge, Decodable'][mouse_i] = np.sum(np.logical_and(decodable, shared_edge_bool))
	edge_cat_mod_counts['Shared Edge, Non-Decodable'][mouse_i] = np.sum(np.logical_and(~decodable, shared_edge_bool))
	
	print(edge_cat_mod_counts['Shared Edge, Decodable'][mouse_i]/(edge_cat_mod_counts['Shared Edge, Decodable'][mouse_i]+edge_cat_mod_counts['Shared Edge, Non-Decodable'][mouse_i]))

	edge_cat_mod_counts['Unique Edge, Decodable'][mouse_i] = np.sum(np.logical_and(decodable, unique_edge_bool))
	edge_cat_mod_counts['Unique Edge, Non-Decodable'][mouse_i] = np.sum(np.logical_and(~decodable, unique_edge_bool))
	
	print(edge_cat_mod_counts['Unique Edge, Decodable'][mouse_i]/(edge_cat_mod_counts['Unique Edge, Decodable'][mouse_i]+edge_cat_mod_counts['Unique Edge, Non-Decodable'][mouse_i]))

	edge_cat_mod_counts['Sign Flip, Decodable'][mouse_i] = np.sum(np.logical_and(decodable, sf_edge_bool))
	edge_cat_mod_counts['Sign Flip, Non-Decodable'][mouse_i] = np.sum(np.logical_and(~decodable, sf_edge_bool))

	print(edge_cat_mod_counts['Sign Flip, Decodable'][mouse_i]/(edge_cat_mod_counts['Sign Flip, Decodable'][mouse_i]+edge_cat_mod_counts['Sign Flip, Non-Decodable'][mouse_i]))
	
	edge_cat_mod_counts['Zero, Decodable'][mouse_i] = np.sum(np.logical_and(decodable, z_edge_bool))
	edge_cat_mod_counts['Zero, Non-Decodable'][mouse_i] = np.sum(np.logical_and(~decodable, z_edge_bool))

	print(edge_cat_mod_counts['Zero, Decodable'][mouse_i]/(edge_cat_mod_counts['Zero, Decodable'][mouse_i]+edge_cat_mod_counts['Zero, Non-Decodable'][mouse_i]))
	
	print('')

	prop_informative[1, mouse_i] = np.sum(np.logical_and(decodable, ~z_edge_bool))/np.sum(~z_edge_bool) # non-zero
	prop_informative[4, mouse_i] = np.sum(np.logical_and(decodable, shared_edge_bool))/np.sum(shared_edge_bool) # shared
	prop_informative[3, mouse_i] = np.sum(np.logical_and(decodable, unique_edge_bool))/np.sum(unique_edge_bool) # unique
	prop_informative[5, mouse_i] = np.sum(np.logical_and(decodable, sf_edge_bool))/np.sum(sf_edge_bool) # sign flip
	prop_informative[2, mouse_i] = np.sum(np.logical_and(decodable, z_edge_bool))/np.sum(z_edge_bool) # zero

	count_informative[1, mouse_i] = np.sum(np.logical_and(decodable, ~z_edge_bool)) # non-zero
	count_informative[4, mouse_i] = np.sum(np.logical_and(decodable, shared_edge_bool)) # shared
	count_informative[3, mouse_i] = np.sum(np.logical_and(decodable, unique_edge_bool)) # unique
	count_informative[5, mouse_i] = np.sum(np.logical_and(decodable, sf_edge_bool)) # sign flip
	count_informative[2, mouse_i] = np.sum(np.logical_and(decodable, z_edge_bool)) # zero

	edge_type_counts[1, mouse_i] = np.sum(~z_edge_bool) # non-zero
	edge_type_counts[4, mouse_i] = np.sum(shared_edge_bool) # shared
	edge_type_counts[3, mouse_i] = np.sum(unique_edge_bool) # unique
	edge_type_counts[5, mouse_i] = np.sum(sf_edge_bool) # sign flip
	edge_type_counts[2, mouse_i] = np.sum(z_edge_bool) # zero


print(edge_cat_mod_counts)

# plot FC decodability by mouse
colors = ['firebrick', 'silver']
fig, ax = plt.subplots()
bottom = np.zeros(len(mice))
for i, (mod, mod_count) in enumerate(mod_counts.items()):
	p = ax.bar(mice, mod_count, label=mod, bottom=bottom, color=colors[i])
	bottom+=mod_count
ax.set_title('Edge Decodability')
ax.legend(loc='upper left')
plt.show()

# plot the proportion of FC that are decodable by mouse
fig, ax = plt.subplots()
bottom = np.zeros(len(mice))
for i, (mod, mod_count) in enumerate(mod_counts.items()):
	p = ax.bar(mice, mod_count/total_edge_counts, label=mod, bottom=bottom, color=colors[i])
	bottom+=mod_count/total_edge_counts
ax.set_title('Proportional Edge Decodability')
ax.legend(loc='upper left')
plt.show()

# plot FC decodability by FC type by mouse
colors = ['maroon', 'firebrick', 'indianred', 'lightcoral', 'black', 'dimgrey', 'darkgrey', 'lightgrey']
fig, ax = plt.subplots()
bottom = np.zeros(len(mice))
for i, (mod, mod_count) in enumerate(edge_cat_mod_counts.items()):
	p = ax.bar(mice, mod_count, label=mod, bottom=bottom, color=colors[i])
	bottom+=mod_count
ax.set_title('Edge Decodability')
ax.legend(loc='upper left')
plt.show()

# plot proportion FC decodability by FC type by mouse
fig, ax = plt.subplots()
bottom = np.zeros(len(mice))
for i, (mod, mod_count) in enumerate(edge_cat_mod_counts.items()):
	p = ax.bar(mice, mod_count/total_edge_counts, label=mod, bottom=bottom, color=colors[i])
	bottom+=mod_count/total_edge_counts
ax.set_title('Proportional Edge Decodability')
ax.legend(loc='upper left')
plt.show()


# plot the proportion of decodable FCs by FC type overall (across mice)
colors = ['k', 'k', 'silver', 'royalblue', 'rebeccapurple', 'mediumorchid']
prop_informative_avg = np.mean(prop_informative, axis=1)
prop_informative_sem = stats.sem(prop_informative, axis=1)
fig, ax = plt.subplots()
ax.bar([4, 5.5, 7], prop_informative_avg[3:], color=colors[3:], yerr=prop_informative_sem[3:], capsize=5, alpha=0.7)

# plot individual points for the proportion decodable by FC type for each mouse
for edge_type_i, edge_type in enumerate(edge_types[3:]):
        prop = np.sort(prop_informative[edge_type_i+3, :])
        offset_max=0.05
        if any(np.diff(prop)<.02):
            left=True
            for d, dot in enumerate(prop):
                r = np.abs(np.random.normal()*offset_max)
                if left:
                    ax.scatter([4, 5.5, 7][edge_type_i]-r, dot, color=colors[edge_type_i+3])
                    left=False
                else:
                    ax.scatter([4, 5.5, 7][edge_type_i]+r, dot, color=colors[edge_type_i+3])
                    left=True
        else:
            ax.scatter(np.ones(len(mice), dtype=object)*[4, 5.5, 7][edge_type_i], prop, color=colors[edge_type_i+3])

 # do t-tests
unique_vs_shared = stats.ttest_rel(prop_informative[3, :], prop_informative[4, :], alternative='two-sided')
unique_vs_sign_flip = stats.ttest_rel(prop_informative[3, :], prop_informative[5, :], alternative='less')
shared_vs_sign_flip = stats.ttest_rel(prop_informative[4, :], prop_informative[5, :], alternative='less')
print(unique_vs_shared)
print(unique_vs_sign_flip)
print(shared_vs_sign_flip)

# add lines for significance
ax.plot([4.05, 5.45], np.ones(2)*np.max(prop_informative[3:4, :])+.02, color='k', ls='-')
if unique_vs_shared.pvalue<.001:
    ax.text(4.75, np.max(prop_informative[2:4, :])+.03, '***', horizontalalignment='center')
elif unique_vs_shared.pvalue<.01:
    ax.text(4.75, np.max(prop_informative[2:4, :])+.03, '**', horizontalalignment='center')
elif unique_vs_shared.pvalue<.05:
    ax.text(4.75, np.max(prop_informative[2:4, :])+.03, '*', horizontalalignment='center')
else:
    ax.text(4.75, np.max(prop_informative[2:4, :])+.03, 'n.s.', horizontalalignment='center')

ax.plot([5.45, 6.95], np.ones(2)*np.max(prop_informative)+.02, color='k', ls='-')
if shared_vs_sign_flip.pvalue<.001:
    ax.text(6.25, np.max(prop_informative)+.025, '***', horizontalalignment='center')
elif shared_vs_sign_flip.pvalue<.01:
    ax.text(6.25, np.max(prop_informative)+.025, '**', horizontalalignment='center')
elif shared_vs_sign_flip.pvalue<.05:
    ax.text(6.25, np.max(prop_informative)+.025, '*', horizontalalignment='center')
else:
    ax.text(6.25, np.max(prop_informative)+.025, 'n.s.', horizontalalignment='center')

ax.plot([4.05, 6.95], np.ones(2)*np.max(prop_informative)+.05, color='k', ls='-')
if unique_vs_sign_flip.pvalue<.001:
    ax.text(5.5, np.max(prop_informative)+.055, '***', horizontalalignment='center')
elif unique_vs_sign_flip.pvalue<.01:
    ax.text(5.5, np.max(prop_informative)+.055, '**', horizontalalignment='center')
elif unique_vs_sign_flip.pvalue<.05:
    ax.text(5.5, np.max(prop_informative)+.055, '*', horizontalalignment='center')
else:
    ax.text(5.5, np.max(prop_informative)+.055, 'n.s.', horizontalalignment='center')


ax.set_ylabel('Proportion of Functional Connections')
ax.set_title('Decodable Functional Connections')
ax.set_xticks([4, 5.5, 7], edge_types[3:])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(fig_save_dir + 'mouse-averaged proportion of informative edges by edge type.pdf', dpi=550)
plt.show()
print(prop_informative_avg[3:], prop_informative_sem[3:])