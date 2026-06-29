'''
this script makes plots of the manually classified carries over time
both for each mouse and the average across all mice
both in raw counts
and each type of carry as a proportion of all carry types

requires (for each mouse and each day of training):
	{mouse}_{day}_classified_pellet_presence_updated.pkl OR {mouse}_{day}_classified_pellet_presence.pkl (pickled dictionary of carry classifications per mouse per training day)
outputs:
	all mice active empty carry counts by type over time.pdf (line plot (+- standard error) of the number of carry trials by type over training)
	all mice active empty carry proportions by type over time.pdf (line plot (+- standard error) of the proportion of carry trials by type over training)
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from scipy import stats

def get_days(mouseID, carry_class_folder):
	classification_files = glob.glob(f'{carry_class_folder}{mouseID}*')
	days = np.unique([file.split('_')[1] for file in classification_files])
	return days

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
carry_classification_folder = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/pellet/'
figures_folder = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/kinematics_3d/general figures/learning/'

drop_counts = []
empty_counts = []
active_counts = []
n_carries = []
n_days = np.zeros(len(mice))
for mouse_i, mouse in enumerate(mice):
	days = get_days(mouse, carry_classification_folder)
	print(days)
	n_days[mouse_i] = len(days)
	empty = np.zeros(len(days))
	drop = np.zeros(len(days))
	active = np.zeros(len(days))
	carry_counter = np.zeros(len(days))
	for i, day in enumerate(days):
		# load in trial labels
		try:
			with open(f'{carry_classification_folder}{mouse}_{day}_classified_pellet_presence_updated.pkl', 'rb') as file:
				data = pickle.load(file)
		except:
			# if a day doesn't have updated labels, use the standard labels
			with open(f'{carry_classification_folder}{mouse}_{day}_classified_pellet_presence.pkl', 'rb') as file:
				data = pickle.load(file)
		drops = 0
		empties = 0
		actives = 0
		for key in data.keys():
			drops += np.sum(data[key]["classifications"]==0)
			empties += np.sum(data[key]["classifications"]==2)
			actives += np.sum(data[key]["classifications"]==1)
			carry_counter[i]+=np.sum(data[key]["classifications"]!=3)
		empty[i] = empties
		drop[i] = drops
		active[i] = actives
	drop_counts.append(drop)
	empty_counts.append(empty)
	active_counts.append(active)
	n_carries.append(carry_counter)

	day_plotting = np.arange(len(days))+1

	# plot active/empty/drop counts
	plt.plot(day_plotting, active, '.-', label='active grasp', color='dodgerblue')
	plt.plot(day_plotting, drop, '.-', label='drops', color='blueviolet')
	plt.plot(day_plotting, empty, '.-', label='empty grasp', color='mediumseagreen')
	plt.legend()
	plt.title(f'{mouse.capitalize()} Carries Over Time')
	plt.xlabel('Training Day')
	plt.ylabel('Count')
	#plt.savefig(figures_folder + f'{mouse} Carries By Type Over Time.png')
	plt.show()

	# plot active/empty/drop proportions
	plt.plot(day_plotting, active/carry_counter, '.-', label='active grasp', color='dodgerblue')
	plt.plot(day_plotting, drop/carry_counter, '.-', label='drops', color='blueviolet')
	plt.plot(day_plotting, empty/carry_counter, '.-', label='empty grasp', color='mediumseagreen')
	plt.legend()
	plt.title(f'{mouse.capitalize()} Carry Proportions Over Time')
	plt.xlabel('Training Day')
	plt.ylabel('Proportion of Carries')
	#plt.savefig(figures_folder + f'{mouse} Carry Proportion By Type Over Time.png')
	plt.show()

min_day = int(min(n_days))
# print(min_day)
# print(n_carries)

# form an array across all mice, aligned by the last day of training
carry_counts = np.zeros((len(mice), min_day, 3))
carry_props = np.zeros((len(mice), min_day, 3))
for mouse_i in range(len(mice)):
	carry_counts[mouse_i, :, 0] = drop_counts[mouse_i][-min_day:]
	carry_counts[mouse_i, :, 1] = active_counts[mouse_i][-min_day:]
	carry_counts[mouse_i, :, 2] = empty_counts[mouse_i][-min_day:]

	carry_props[mouse_i, :, 0] = drop_counts[mouse_i][-min_day:]/n_carries[mouse_i][-min_day:]
	carry_props[mouse_i, :, 1] = active_counts[mouse_i][-min_day:]/n_carries[mouse_i][-min_day:]
	carry_props[mouse_i, :, 2] = empty_counts[mouse_i][-min_day:]/n_carries[mouse_i][-min_day:]

days = np.arange(min_day)+1

# plot the active/empty/drop counts as average+-sem across all mice
mean_counts = np.mean(carry_counts, axis=0)
sem_counts = stats.sem(carry_counts, axis=0)
labels = ['drops', 'active grasp', 'empty grasp']
fig, ax = plt.subplots()
ax.plot(days, mean_counts[:, 2], label=labels[2], color='blueviolet')
ax.fill_between(days, mean_counts[:, 2]+sem_counts[:, 2], mean_counts[:, 2]-sem_counts[:, 2], color='blueviolet', alpha=0.4)
ax.plot(days, mean_counts[:, 1], label=labels[1], color='dodgerblue')
ax.fill_between(days, mean_counts[:, 1]+sem_counts[:, 1], mean_counts[:, 1]-sem_counts[:, 1], color='dodgerblue', alpha=0.4)
ax.plot(days, mean_counts[:, 0], label=labels[0], color='mediumseagreen')
ax.fill_between(days, mean_counts[:, 0]+sem_counts[:, 0], mean_counts[:, 0]-sem_counts[:, 0], color='mediumseagreen', alpha=0.4)
ax.legend()
ax.set_xlabel('Training Day')
ax.set_ylabel('Count')
ax.set_title('Carry Counts Over Time')
ax.set_xticks(np.arange(min_day)+1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig(figures_folder + 'all mice carry counts by type over time.png', dpi=660)
plt.show()

# plot the active/empty counts as average+-sem across all mice
fig, ax = plt.subplots()
ax.plot(days, mean_counts[:, 2], label=labels[2], color='k')
ax.fill_between(days, mean_counts[:, 2]+sem_counts[:, 2], mean_counts[:, 2]-sem_counts[:, 2], color='k', alpha=0.4)
ax.plot(days, mean_counts[:, 1], label=labels[1], color='indianred')
ax.fill_between(days, mean_counts[:, 1]+sem_counts[:, 1], mean_counts[:, 1]-sem_counts[:, 1], color='indianred', alpha=0.4)
#ax.legend()
ax.set_xlabel('Training Day')
ax.set_ylabel('Count')
ax.set_title('Carry Counts Over Time')
ax.set_xticks(np.arange(min_day)+1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim((days[0], days[-1]))
plt.savefig(figures_folder + 'all mice active empty carry counts by type over time.pdf', dpi=660)
plt.show()

# plot the empty counts as average+-sem across all mice
mean_counts = np.mean(carry_counts[:, :, 2], axis=0)
fig, ax = plt.subplots()
ax.errorbar(days, mean_counts, yerr=stats.sem(carry_counts[:, :, 2], axis=0), color='cornflowerblue', marker='.', ecolor='cornflowerblue', capsize=3)
ax.set_xlabel('Training Day')
ax.set_xticks(np.arange(min_day)+1)
ax.set_ylabel('Count')
ax.set_title('Empty Grasp Trial Counts Over Time')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim((days[0], days[-1]))
#plt.savefig(figures_folder + 'all mice empty grasp carry counts over time.png', dpi=660)
plt.show()

# plot the active/empty/drop proportions as average+-sem across all mice
mean_props = np.nanmean(carry_props, axis=0)
sem_props = stats.sem(carry_props, axis=0, nan_policy='omit')
fig, ax = plt.subplots()
ax.plot(days, mean_props[:, 2], label=labels[2], color='blueviolet')
ax.fill_between(days, mean_props[:, 2]+sem_props[:, 2], mean_props[:, 2]-sem_props[:, 2], color='blueviolet', alpha=0.4)
ax.plot(days, mean_props[:, 1], label=labels[1], color='dodgerblue')
ax.fill_between(days, mean_props[:, 1]+sem_props[:, 1], mean_props[:, 1]-sem_props[:, 1], color='dodgerblue', alpha=0.4)
ax.plot(days, mean_props[:, 0], label=labels[0], color='mediumseagreen')
ax.fill_between(days, mean_props[:, 0]+sem_props[:, 0], mean_props[:, 0]-sem_props[:, 0], color='mediumseagreen', alpha=0.4)
ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Training Day')
ax.set_ylabel('Proportion of Carries')
ax.set_xticks(np.arange(min_day)+1)
ax.set_title('Proportions of Carries By Type Over Time')
ax.set_xlim((days[0], days[-1]))
#plt.savefig(figures_folder + 'all mice carry proportions by type over time.png', dpi=660)
plt.show()

# plot the active/empty proportions as average+-sem across all mice
fig, ax = plt.subplots()
ax.plot(days, mean_props[:, 2], label=labels[2], color='k')
ax.fill_between(days, mean_props[:, 2]+sem_props[:, 2], mean_props[:, 2]-sem_props[:, 2], color='k', alpha=0.4)
ax.plot(days, mean_props[:, 1], label=labels[1], color='indianred')
ax.fill_between(days, mean_props[:, 1]+sem_props[:, 1], mean_props[:, 1]-sem_props[:, 1], color='indianred', alpha=0.4)
#ax.legend()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Training Day')
ax.set_ylabel('Proportion of Carries')
ax.set_xticks(np.arange(min_day)+1)
ax.set_title('Proportions of Carries By Type Over Time')
ax.set_xlim((days[0], days[-1]))
plt.savefig(figures_folder + 'all mice active empty carry proportions by type over time.pdf', dpi=660)
plt.show()