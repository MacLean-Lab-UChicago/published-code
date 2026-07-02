'''
plot reward counts for each mouse, averaged across mice
and success rates for each mouse and averaged across mice
'''
import numpy as np
import src.IO
import src.utils
import matplotlib.pyplot as plt
from scipy.stats import sem
from pathlib import Path
data_dir = Path.cwd() / 'data'

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
training_days = np.arange(12)+1
multi_mouse_save_dir = data_dir + '/results/cross_mouse_results/learning/'

cross_mouse_success = np.zeros((len(training_days), len(mice)))*np.nan
cross_mouse_rate = np.zeros((len(training_days), len(mice)))*np.nan
for mouse_i, mouseID in enumerate(mice):
	days = src.IO.get_days(mouseID)
	drive = data_dir + '/neural'
	mouse_dir = drive + '/' + mouseID + '/'
	fig_save_dir = mouse_dir + 'carry_analysis/learning/figures/'
	pellet_dir = f'{data_dir}/kinematics/{mouseID}/pellet_presence/'

	success = np.zeros(len(training_days))*np.nan
	rate = np.zeros(len(training_days))*np.nan
	day_counter = -1
	for day in days:
		s2p_fld = src.IO.get_s2p_fld(mouseID, day)
		try:
			# load in success/fail labels
			s = np.load(s2p_fld + 'success_fail_bool.npy')
			day_counter+=1
		except:
			#print(f'{day} not a training, day, continuing...')
			if day_counter>0:
				day_counter+=1
			continue
		# load in if a pellet was present for each pellet delivery
		pellet_presence = src.IO.load_pickle(pellet_dir + f'{mouseID}_{day}_pellet_frames_and_presence_dict.pkl')
		# count the number of pellet deliveries
		pellets = []
		for event in pellet_presence.keys():
			pellets.extend(pellet_presence[event]['pellet_present'])
		# count the number of successes
		success[day_counter] = np.sum(s)
		# divide successes by pellet deliveries to get success rate
		rate[day_counter] = np.sum(s)/np.sum(pellets)
	cross_mouse_success[:, mouse_i] = success
	cross_mouse_rate[:, mouse_i] = rate

	# plot success count per mouse
	plt.plot(training_days, success, color='k')
	plt.xlabel('Training Day')
	plt.ylabel('Number of Successful Trials')
	plt.title(mouseID)
	plt.savefig(fig_save_dir + f'{mouseID} number of successes.png')
	plt.close()

	# plot success rate per mouse
	plt.plot(training_days, rate, color='k')
	plt.xlabel('Training Day')
	plt.ylabel('Success Rate (%)')
	plt.title(mouseID)
	plt.savefig(fig_save_dir + f'{mouseID} success rate.png')
	plt.close()

# find avg success count and sem across mice
avg_success_count = np.nanmean(cross_mouse_success, axis=1)
success_error = sem(cross_mouse_success, axis=1, nan_policy = 'omit')
# and plot
fig, ax = plt.subplots()
ax.plot(training_days, avg_success_count, color='k')
ax.fill_between(training_days, avg_success_count-success_error, avg_success_count+success_error, color='k', alpha=.4)
ax.set_xlabel('Training Day')
ax.set_ylabel('Number of Successful Trials')
ax.set_xlim([1,11])
ax.set_xticks(np.arange(11)+1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(multi_mouse_save_dir + 'number of successes.pdf', dpi=550)
plt.show()

# find avg success rate and sem across mice
avg_success_rate = np.nanmean(cross_mouse_rate, axis=1)
rate_error = sem(cross_mouse_rate, axis=1, nan_policy = 'omit')
# and plot
fig, ax = plt.subplots()
ax.plot(training_days, avg_success_rate, color='k')
ax.fill_between(training_days, avg_success_rate-rate_error, avg_success_rate+rate_error, color='k', alpha=.4)
ax.set_xlabel('Training Day')
ax.set_ylabel('Success Rate (%)')
ax.set_xlim([1,11])
ax.set_xticks(np.arange(11)+1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(multi_mouse_save_dir + 'success rate.pdf', dpi=550)
plt.show()

#plot success rates and counts on the same plot with different y axes
fig, ax1 = plt.subplots()

# plot success rate on the primary (left) y-axis
color1 = 'xkcd:forest green'
ax1.set_xlabel('Training Day')
ax1.set_ylabel('Success Rate (%)', color=color1)
ax1.plot(training_days, avg_success_rate, color=color1)
ax1.fill_between(training_days, avg_success_rate-rate_error, avg_success_rate+rate_error, color=color1, alpha=.2)
ax1.tick_params(axis='y', labelcolor=color1)

ax1.set_xlim([1,11])
ax1.set_xticks(np.arange(11)+1)
ax1.set_ylim((0, max(avg_success_rate+rate_error)))

# plot success count on the secondary (right) y-axis
ax2 = ax1.twinx()  # Shared x-axis, independent y-axis
color2 = 'xkcd:royal blue'
ax2.set_ylabel('Number of Successful Trials', color=color2)
ax2.plot(training_days, avg_success_count, color=color2)
ax2.fill_between(training_days, avg_success_count-success_error, avg_success_count+success_error, color=color2, alpha=.2)
ax2.tick_params(axis='y', labelcolor=color2)

ax2.set_ylim((0, max(avg_success_count+success_error)))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)

fig.suptitle('Learning Over Time')

fig.tight_layout()  # Ensures the right label isn't clipped
fig.savefig(multi_mouse_save_dir + 'success rate and count overlayed.pdf', dpi=550)
plt.show()