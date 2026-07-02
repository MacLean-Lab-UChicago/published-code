import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

categories = ['non-decodable\ninstant fr', 'decodable\ninstant fr', 'full pop\ninstant fr', 'non-decodable\navg fr', 'decodable\navg fr', 'full pop\navg fr']
save_path = data_dir + '/results/cross_mouse_results/'
fig_savepath = save_path + 'figures/'
test_size = 0.1
control=False
reorder=False
instant_fr=False

if control:
    accuracy = np.load(save_path + f'Control_SVM_accuracies_test_{test_size}.npy') 
    errors = np.load(save_path + f'Control_SVM_error_test_{test_size}.npy')
    CI = np.load(save_path + f'Control_95_CI_test_{test_size}.npy')
else:
    accuracy = np.load(save_path + f'SVM_accuracies_test_{test_size}.npy')
    errors = np.load(save_path + f'SVM_error_test_{test_size}.npy')
    CI = np.load(save_path + f'95_CI_test_{test_size}.npy')


if reorder: # just for visualizations
    for i in range(len(mice)):
        accuracy[i] = accuracy[i][[0, 3, 1, 4, 2, 5]]
        errors[i] = errors[i][[0, 3, 1, 4, 2, 5]]

# Bar settings
n_groups = len(mice)
n_categories = len(categories)
bar_width = .15
index = np.arange(n_groups)
colors = ['#ADE2FF', '#83C9F4', '#31A5ED', '#AEE0B1', '#6AAC61', '#008F43']
if reorder:
    colors = np.array(colors)[[0, 3, 1, 4, 2, 5]]
    categories = np.array(categories, dtype=object)[[0, 3, 1, 4, 2, 5]]

# Plot
fig, ax = plt.subplots()

for i in range(n_categories):
    ax.bar(
        index + i * bar_width,
        accuracy[:, i],
        bar_width,
        yerr=errors[:, i],
        label=categories[i],
        capsize=5,
        color=colors[i]
    )

# Labels and legend
#ax.set_xlabel('Mouse')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.35, 1])
ax.axhline(0.5, color='k', ls='--', label='chance')
ax.set_title('Accuracy for Neural SVM, Active Grasp vs No Pellet')
ax.set_xticks(index + n_categories/2*bar_width-bar_width/2)
ax.set_xticklabels(mice)
ax.legend()

#plt.tight_layout()
plt.show()

# compare neural to kinematic
kin_dir =  data_dir + '/results/cross_mouse_results/'
if instant_fr:
    neural_accuracy = np.load(save_path + f'SVM_accuracies_test_{test_size}.npy')[:, 2]
    neural_error = np.load(save_path + f'SVM_error_test_{test_size}.npy')[:, 2]
else:
    neural_accuracy = np.load(save_path + f'SVM_accuracies_test_{test_size}.npy')[:, 5]
    neural_error = np.load(save_path + f'SVM_error_test_{test_size}.npy')[:, 5]

# gross fine and all kinematics compared to neural, mouse-averaged
gross_accuracies = np.load(kin_dir + 'gross_SVM_accuracies.npy')
fine_accuracies = np.load(kin_dir + 'fine_SVM_accuracies.npy')
kin_accuracies = np.load(kin_dir + 'kin_SVM_accuracies.npy')

mean_neural = np.mean(neural_accuracy)
mean_gross = np.mean(gross_accuracies)
mean_fine = np.mean(fine_accuracies)
mean_kin = np.mean(kin_accuracies)

print(mean_gross, mean_fine, mean_kin, mean_neural)

sem_neural = stats.sem(neural_accuracy)
sem_gross = stats.sem(gross_accuracies)
sem_fine = stats.sem(fine_accuracies)
sem_kin = stats.sem(kin_accuracies)

print(sem_gross, sem_fine, sem_kin, sem_neural)

by_mouse_a = np.concatenate([gross_accuracies[:, np.newaxis], fine_accuracies[:, np.newaxis], kin_accuracies[:, np.newaxis], neural_accuracy[:, np.newaxis]], axis=1)

colors = ['cornflowerblue', 'royalblue', 'mediumblue', 'mediumseagreen']
categories = ['gross kinematics', 'fine kinematics', 'all kinematics', 'neural activity']

fig, ax = plt.subplots()
ax.bar([0, 2, 4, 6], [mean_gross, mean_fine, mean_kin, mean_neural], width=1.5, color=colors, alpha=0.5, yerr=[sem_gross, sem_fine, sem_kin, sem_neural], capsize=5)
ax.set_xticks([0, 2, 4, 6], categories)
for cat_i, category in enumerate(categories):
    accur = np.sort(by_mouse_a[:, cat_i])
    offset_max=0.2
    if any(np.diff(accur)<.02):
        left=True
        for d, dot in enumerate(accur):
            r = np.random.rand()*offset_max
            if left:
                ax.scatter([0, 2, 4, 6][cat_i]-r, dot, color=colors[cat_i])
                left=False
            else:
                ax.scatter([0, 2, 4, 6][cat_i]+r, dot, color=colors[cat_i])
                left=True
    else:
        ax.scatter(np.ones(len(mice), dtype=object)*[0, 2, 4, 6][cat_i], accur, color=colors[cat_i])

ax.set_ylim([0.35, 1])
ax.axhline(0.5, color='k', ls='--', label='chance')
ax.set_ylabel('decoding accuracy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# significance testing
# do t-tests
neural_vs_gross = stats.ttest_rel(neural_accuracy, gross_accuracies, alternative='greater')
neural_vs_fine = stats.ttest_rel(neural_accuracy, fine_accuracies, alternative='greater')
neural_vs_all = stats.ttest_rel(neural_accuracy, kin_accuracies, alternative='greater')
print(neural_vs_gross)
print(neural_vs_fine)
print(neural_vs_all)

# add lines for significance
ax.plot([4.05, 5.95], np.ones(2)*np.max(by_mouse_a)+.02, color='k', ls='-')
if neural_vs_gross.pvalue<.001:
    ax.text(5, np.max(by_mouse_a)+.025, '***', horizontalalignment='center')
elif neural_vs_gross.pvalue<.01:
    ax.text(5, np.max(by_mouse_a)+.025, '**', horizontalalignment='center')
elif neural_vs_gross.pvalue<.05:
    ax.text(5, np.max(by_mouse_a)+.025, '*', horizontalalignment='center')
else:
    ax.text(5, np.max(by_mouse_a)+.025, 'n.s.', horizontalalignment='center')

ax.plot([2.05, 5.95], np.ones(2)*np.max(by_mouse_a)+.05, color='k', ls='-')
if neural_vs_fine.pvalue<.001:
    ax.text(4, np.max(by_mouse_a)+.055, '***', horizontalalignment='center')
elif neural_vs_fine.pvalue<.01:
    ax.text(4, np.max(by_mouse_a)+.055, '**', horizontalalignment='center')
elif neural_vs_fine.pvalue<.05:
    ax.text(4, np.max(by_mouse_a)+.055, '*', horizontalalignment='center')
else:
    ax.text(4, np.max(by_mouse_a)+.055, 'n.s.', horizontalalignment='center')

ax.plot([.05, 5.95], np.ones(2)*np.max(by_mouse_a)+.08, color='k', ls='-')
if neural_vs_all.pvalue<.001:
    ax.text(3, np.max(by_mouse_a)+.085, '***', horizontalalignment='center')
elif neural_vs_all.pvalue<.01:
    ax.text(3, np.max(by_mouse_a)+.085, '**', horizontalalignment='center')
elif neural_vs_all.pvalue<.05:
    ax.text(3, np.max(by_mouse_a)+.085, '*', horizontalalignment='center')
else:
    ax.text(3, np.max(by_mouse_a)+.085, 'n.s.', horizontalalignment='center')

if instant_fr:
    plt.savefig(f'{fig_savepath}mouse_avg_kin_vs_neural_instant_fr.pdf', dpi=500)
else:
    plt.savefig(f'{fig_savepath}mouse_avg_kin_vs_neural_avg_fr.pdf', dpi=500)
plt.show()

# average across mice
accuracies = np.mean(neural_accuracy) # average across mice
errors = stats.sem(neural_accuracy)
kin_accuracies = np.mean(kin_accuracy) # average across mice
kin_errors = stats.sem(kin_accuracy)
print(accuracies, errors)
print(kin_accuracies, kin_errors)

plt.bar(['neural', 'kinematic'], [accuracies, kin_accuracies], yerr = [errors, kin_errors])
plt.show()


# combined across mice
if control:
    accuracy = np.load(save_path + f'Control_SVM_accuracies_test_{test_size}.npy') 
    errors = np.load(save_path + f'Control_SVM_error_test_{test_size}.npy')
    CI = np.load(save_path + f'Control_95_CI_test_{test_size}.npy')
else:
    accuracy = np.load(save_path + f'SVM_accuracies_test_{test_size}.npy')
    errors = np.load(save_path + f'SVM_error_test_{test_size}.npy')
    CI = np.load(save_path + f'95_CI_test_{test_size}.npy')
accuracies = np.mean(accuracy, axis=0) # average across mice
errors = stats.sem(accuracy, axis=0)
print(accuracies)
print(errors)
if instant_fr:
    a = np.append(accuracies[:3], mean_kin)
    e = np.append(errors[:3], sem_kin)
    by_mouse_a = np.concatenate([accuracy[:, :3], kin_accuracies[:, np.newaxis]], axis=1)
else:
    a = np.append(accuracies[3:], mean_kin)
    e = np.append(errors[3:], sem_kin)
    by_mouse_a = np.concatenate([accuracy[:, 3:], kin_accuracies[:, np.newaxis]], axis=1)

# Bar settings
n_groups = 1
n_categories = 3
index = np.arange(n_groups)
colors = ['darkgrey', 'indianred', '#7d5555ff', 'k']
categories = ['non-decodable cells', 'decodable cells', 'full population', 'kinematics']

# do t-tests
non_decode_vs_decode = stats.ttest_rel(by_mouse_a[:, 0], by_mouse_a[:, 1], alternative='less')
non_decode_vs_full_pop = stats.ttest_rel(by_mouse_a[:, 0], by_mouse_a[:, 2], alternative='less')
decode_vs_full_pop = stats.ttest_rel(by_mouse_a[:, 1], by_mouse_a[:, 2], alternative='two-sided')
full_pop_vs_kin = stats.ttest_rel(by_mouse_a[:,2], kin_accuracies, alternative='greater')
print(non_decode_vs_decode)
print(non_decode_vs_full_pop)
print(decode_vs_full_pop)
print(full_pop_vs_kin)

print('')

# one sample t-tests
non_decode = stats.ttest_1samp(by_mouse_a[:, 0], 0.5, alternative='greater')
full_pop = stats.ttest_1samp(by_mouse_a[:, 2], 0.5, alternative='greater')
decode = stats.ttest_1samp(by_mouse_a[:, 1], 0.5, alternative='greater')
kin = stats.ttest_1samp(kin_accuracies, 0.5, alternative='greater')
print(non_decode, non_decode.confidence_interval())
print(full_pop)
print(decode)
print(kin)

# Plot
fig, ax = plt.subplots()

ax.bar([2.5, 4, 5.5, 7], a, yerr=e, capsize=5, color=colors, alpha=0.5)
for cat_i, category in enumerate(categories):
    accur = np.sort(by_mouse_a[:, cat_i])
    offset_max=0.1
    if any(np.diff(accur)<.02):
        left=True
        for d, dot in enumerate(accur):
            r = np.random.rand()*offset_max
            if left:
                ax.scatter([2.5, 4, 5.5, 7][cat_i]-r, dot, color=colors[cat_i])
                left=False
            else:
                ax.scatter([2.5, 4, 5.5, 7][cat_i]+r, dot, color=colors[cat_i])
                left=True
    else:
        ax.scatter(np.ones(len(mice), dtype=object)*[2.5, 4, 5.5][cat_i], accur, color=colors[cat_i])

# add lines for significance
ax.plot([2.55, 3.95], np.ones(2)*np.max(by_mouse_a)+.02, color='k', ls='-')
if non_decode_vs_decode.pvalue<.001:
    ax.text(3.25, np.max(by_mouse_a)+.025, '***', horizontalalignment='center')
elif non_decode_vs_decode.pvalue<.01:
    ax.text(3.25, np.max(by_mouse_a)+.025, '**', horizontalalignment='center')
elif non_decode_vs_decode.pvalue<.05:
    ax.text(3.25, np.max(by_mouse_a)+.025, '*', horizontalalignment='center')
else:
    ax.text(3.25, np.max(by_mouse_a)+.025, 'n.s.', horizontalalignment='center')

ax.plot([2.55, 5.45], np.ones(2)*np.max(by_mouse_a)+.06, color='k', ls='-')
if non_decode_vs_full_pop.pvalue<.001:
    ax.text(4, np.max(by_mouse_a)+.065, '***', horizontalalignment='center')
elif non_decode_vs_full_pop.pvalue<.01:
    ax.text(4, np.max(by_mouse_a)+.065, '**', horizontalalignment='center')
elif non_decode_vs_full_pop.pvalue<.05:
    ax.text(4, np.max(by_mouse_a)+.065, '*', horizontalalignment='center')
else:
    ax.text(4, np.max(by_mouse_a)+.065, 'n.s.', horizontalalignment='center')

ax.plot([4.05, 5.45], np.ones(2)*np.max(by_mouse_a)+.02, color='k', ls='-')
if decode_vs_full_pop.pvalue<.001:
    ax.text(4.75, np.max(by_mouse_a)+.03, '***', horizontalalignment='center')
elif decode_vs_full_pop.pvalue<.01:
    ax.text(4.75, np.max(by_mouse_a)+.03, '**', horizontalalignment='center')
elif decode_vs_full_pop.pvalue<.05:
    ax.text(4.75, np.max(by_mouse_a)+.03, '*', horizontalalignment='center')
else:
    ax.text(4.75, np.max(by_mouse_a)+.03, 'n.s.', horizontalalignment='center')

ax.plot([5.55, 6.95], np.ones(2)*np.max(by_mouse_a)+.02, color='k', ls='-')
if full_pop_vs_kin.pvalue<.001:
    ax.text(6.25, np.max(by_mouse_a)+.025, '***', horizontalalignment='center')
elif full_pop_vs_kin.pvalue<.01:
    ax.text(6.25, np.max(by_mouse_a)+.025, '**', horizontalalignment='center')
elif full_pop_vs_kin.pvalue<.05:
    ax.text(6.25, np.max(by_mouse_a)+.025, '*', horizontalalignment='center')
else:
    ax.text(6.25, np.max(by_mouse_a)+.025, 'n.s.', horizontalalignment='center')

# Labels and legend
ax.set_ylabel('Accuracy')
ax.set_ylim([0.35, 1])
ax.axhline(0.5, color='k', ls='--', label='chance')
ax.set_xticks([2.5, 4, 5.5, 7], categories)
#ax.set_title('Accuracy for Neural SVM, Active vs Empty Grasp')
#ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.tight_layout()
if instant_fr:
    plt.savefig(f'{fig_savepath}neural_pop_decoding_instant_fr.pdf', dpi=500)
else:
    plt.savefig(f'{fig_savepath}neural_pop_decoding_avg_fr.pdf', dpi=500)
plt.show()
