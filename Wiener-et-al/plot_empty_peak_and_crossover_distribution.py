'''
plots a cdf of relative peak and crossover days for all mice
'''
import src.IO
import numpy as np
import matplotlib.pyplot as plt

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

relative_peak_days = np.zeros(len(mice))*np.nan
relative_cross_days = np.zeros(len(mice))*np.nan
m46_late=False

for i, mouseID in enumerate(mice):
	# get peak and crossover days
	peak_day = src.IO.get_carry_peak(mouseID, m46_late)
	cross_day = src.IO.get_carry_crossover(mouseID, m46_late)
	# compare to training days to get relative training day for each event
	days = src.IO.get_days(mouseID)
	first_day = src.IO.get_first_day(mouseID)
	print(np.where(np.array(days, dtype=object)==first_day)[0][0])
	days = np.array(days, dtype=object)[np.where(np.array(days, dtype=object)==first_day)[0][0]:]

	# convert to 1-indexing from 0-indexing
	relative_peak_days[i] = np.where(days==peak_day)[0] + 1
	relative_cross_days[i] = np.where(days==cross_day)[0] + 1

# find counts of mice for each day where there is a crossover or empty peak
cross_days, cross_counts = np.unique(relative_cross_days, return_counts=True)
peak_days, peak_counts = np.unique(relative_peak_days, return_counts=True)

print(cross_days, cross_counts)
print(peak_days, peak_counts)

# plot in the style of a timeline
fig, ax = plt.subplots(figsize=(7, 2))
# plot points where there is at least one instance of a peak or crossover
# crossovers slightly elevated compared to the points indicating peaks for separation on days where there are both
ax.scatter(peak_days, np.ones(len(peak_days))*.1, color='orange', label='empty grasp peak day')
ax.scatter(cross_days, np.ones(len(cross_days))*.105, color='coral', label='active-empty crossover day')
# for days with more than one instance of a peak, add points until the total number is plotted (histogram-style)
for peak, count in zip(peak_days, peak_counts):
	counter=1
	location=.11
	while count>counter:
		ax.scatter(peak, location, color='orange')
		counter+=1
		location+=.01
# repeat for crossovers with the slight vertical offset compared to empty peak instances
for cross, count in zip(cross_days, cross_counts):
	counter=1
	location=0.115
	while count>counter:
		ax.scatter(cross, location, color='coral')
		counter+=1
		location+=.01
# add lines indicating the average of the distribution across mice
# this information is from plotting the total trial counts across all mice
ax.axvline(3, color='orange', alpha=0.5, lw=2, label='mouse-averaged peak day')
ax.axvline(5, color='coral', alpha=0.5, lw=2, label='mouse-averaged crossover day')
ax.legend()
ax.set_xticks(np.arange(len(days))+1)
ax.set_xlabel('Training Day')
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_yticks([])
ax.set_ylim((.09, .18))
plt.tight_layout()
plt.savefig('/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/kinematics_3d/general figures/learning/peak and crossover timeline.pdf', dpi=500)
plt.show()
