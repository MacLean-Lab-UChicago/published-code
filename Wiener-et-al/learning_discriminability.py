'''
d prime analysis for discriminability
'''
import numpy as np
import src.IO
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
data_dir = Path.cwd() / 'data'

def d_prime_calc(a, b, std_a, std_b):
	d = np.abs(a-b)/np.sqrt(.5*(np.square(std_a)+np.square(std_b)))
	return d

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']

multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/learning/'
t_pre=4 # frames
t_post=4 # frames
time_in_trial = t_pre+t_post # frames
time = (np.arange(-t_pre, t_post))*33 # ms
calculate = True
day_pairs = True # use pairs of days instead of single days
m46_late = False
days_rel_to = ['peak', 'crossover']

if day_pairs:
	for rel_to in days_rel_to:
		rel_diff = [[] for mouse in mice]
		for mouse_i, mouseID in enumerate(mice):
			drive = data_dir + '/neural'
			mouse_dir = drive + '/' + mouseID + '/'
			if rel_to=='peak':
				learning_days = src.IO.get_carry_peak_learning_days(mouseID, m46_late)
			else: # if relative to crossover
				learning_days = src.IO.get_carry_thresh_learning_days(mouseID, m46_late)
			save_dir = mouse_dir + '/carry_analysis/learning/'
			fig_dir = save_dir + 'figures/'
			
			if learning_days==None:
				continue
			
			day_pairs = [f'{learning_days[d]}-{learning_days[d+1]}' for d in range(len(learning_days)-1)]
			print(mouseID)
			print(learning_days)

			if calculate:
				reg_inds = src.utils.load_registered_cells(mouse_dir, learning_days)
				
				# grab the neural activity per day over all learning days
				act_spks = [[] for day in learning_days]
				dry_spks = [[] for day in learning_days]
				day_labels = {'active':[], 'empty':[]}
				for i, day in enumerate(learning_days):
					# load in the Cascade spikes
					s2p_fld = src.IO.get_s2p_fld(mouseID, day)
					
					spks = np.load(s2p_fld + 'cascade_spks.npy')
					# index by the registered cells
					reg_spks = spks[reg_inds[i], :]
					n_neurons=reg_spks.shape[0]

					# load in the behavioral times
					carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
					carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
					s_carry_times = carry_times[carry_labels==1]
					dry_carry_times = carry_times[carry_labels==2]

					print(len(s_carry_times), len(dry_carry_times))

					# grab relevent neural data to the behavioral times
					day_labels['active'].extend(np.ones(len(s_carry_times))*i)
					day_labels['empty'].extend(np.ones(len(dry_carry_times))*i)
					temp_a_spks = np.zeros((n_neurons, len(s_carry_times), time_in_trial))
					temp_d_spks = np.zeros((n_neurons, len(dry_carry_times), time_in_trial))
					for carry_i, carry in enumerate(s_carry_times.astype(int)):
						print(carry, reg_spks.shape)
						temp_a_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					for carry_i, carry in enumerate(dry_carry_times.astype(int)):
						temp_d_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					act_spks[i]=temp_a_spks
					dry_spks[i]=temp_d_spks


				d_prime = np.zeros((n_neurons, len(day_pairs)))
				for pair_i, pair in enumerate(day_pairs):
					# concatenate over the pair of days
					a_spks = np.concatenate((act_spks[pair_i], act_spks[pair_i+1]), axis=1)
					d_spks = np.concatenate((dry_spks[pair_i], dry_spks[pair_i+1]), axis=1)

					# take the firing rate averaged across time per trial
					time_avg_a = np.nanmean(a_spks, axis=2)*30
					time_avg_d = np.nanmean(d_spks, axis=2)*30

					# find the mean and standard deviation firing rate across trials
					trial_time_avg_a = np.nanmean(time_avg_a, axis=1)
					trial_time_avg_d = np.nanmean(time_avg_d, axis=1)
					trial_time_std_a = np.nanstd(time_avg_a, axis=1, ddof=1) # ddof = 1 for unbiased standard deviation metric
					trial_time_std_d = np.nanstd(time_avg_d, axis=1, ddof=1)

					# calculate d prime
					d = d_prime_calc(trial_time_avg_a, trial_time_avg_d, trial_time_std_a, trial_time_std_d)
					
					d_prime[:, pair_i] = d
					assert len(d)==n_neurons, 'took d prime over wrong axes'
					print('full pop: ', np.nanmean(d), '+-', stats.sem(d, nan_policy='omit'))
					print('range: ', np.nanmin(d), np.nanmax(d))
			
			
				if mouseID=='mouse46' and m46_late:
					np.save(save_dir+f'day_pair_d_prime_late_days_rel_to_{rel_to}.npy', d_prime)
				else:
					np.save(save_dir+f'day_pair_d_prime_diff_rel_to_{rel_to}.npy', d_prime)

			else:
				if mouseID=='mouse46' and m46_late:
					d_prime = np.load(save_dir+f'day_pair_d_prime_late_days_rel_to_{rel_to}.npy')
				else:
					d_prime = np.load(save_dir+f'day_pair_d_prime_diff_rel_to_{rel_to}.npy')
				n_neurons = len(d_prime)

			for pair_i, day_pair in enumerate(day_pairs):
				# print the number of cells with d' higher than .3
				print(np.sum(d_prime[:, pair_i]>.3), np.sum(d_prime[:, pair_i]>.3)/d_prime.shape[0])

			# plot d' across the full population per pair of days per mouse
			pop_d = np.nanmean(d_prime, axis=0)
			pop_sem = stats.sem(d_prime, axis=0, nan_policy='omit')

			plt.errorbar(day_pairs, pop_d, yerr=pop_sem, color='k')
			plt.ylabel("Average d'")
			plt.xlabel('Day Pair')
			plt.title(f'{mouseID} Active-Empty D-Prime')
			if mouseID=='mouse46' and m46_late:
				plt.savefig(fig_dir + f'Active-Empty D Prime Over Late Paired Days rel to {rel_to}.png')
			else:
				plt.savefig(fig_dir + f'Active-Empty D Prime Over Paired Days rel to {rel_to}.png')
			plt.close()

			if rel_to=='peak':
				day_of_interest = src.IO.get_carry_peak(mouseID, m46_late)
			else:
				day_of_interest = src.IO.get_carry_threshold_cross(mouseID, m46_late)

			rel_doi = int(np.where(np.array(learning_days, dtype=object)==day_of_interest)[0])
			print(rel_doi)
			rel_diff_mouse = np.zeros((n_neurons, 10))*np.nan
			rel_diff_mouse[:, (3-rel_doi):(3-rel_doi+4)] = d_prime
			rel_diff[mouse_i] = rel_diff_mouse

		# concatenate across mice
		rel_diff = np.concatenate(rel_diff, axis=0)

		# plot d' across the full population (all mice) per pair of days relative to peak/crossover
		pop_diff = np.nanmean(rel_diff, axis=0)
		pop_sem = stats.sem(rel_diff, axis=0, nan_policy='omit')
		fig, ax = plt.subplots()
		ax.plot([-3, -2, -1, 0, 1], pop_diff[:5], color='darkred')
		ax.fill_between([-3, -2, -1, 0, 1], pop_diff[:5]+pop_sem[:5], pop_diff[:5]-pop_sem[:5], color='darkred', alpha=.4)
		ax.axvline(-.5, color='k', ls='--')
		ax.set_ylabel("Mean d'")
		if rel_to=='peak':
			ax.set_xlabel('Training Day Pair from Empty Grasp Peak')
		else:
			ax.set_xlabel('Training Day Pair from Active-Empty Crossover')
		ax.set_title('Active-Empty Discriminability Over Learning')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_xticks([-2, -1, 0, 1], labels=['(-2,-1)', '(-1,0)', '(0,1)', '(1,2)'])
		ax.set_xlim(([-2, 1]))
		if m46_late:
			plt.savefig(multimouse_fig_dir + f'Day Pair Active-Empty d prime Rel to {rel_to} m46 late days.pdf', dpi=500)
		else:
			plt.savefig(multimouse_fig_dir + f'Day Pair Active-Empty d prime Rel to {rel_to}.pdf', dpi=500)
		plt.show()

		result = stats.ttest_rel(rel_diff[:, 1:3].flatten(), rel_diff[:, 3:5].flatten(), nan_policy='omit', alternative='less')
		print(result)
		print(np.nanmean(rel_diff[:, 1:3].flatten()), np.nanmean(rel_diff[:, 3:5].flatten()))


else: # single days
	for rel_to in days_rel_to:
		rel_diff = [[] for mouse in mice]
		for mouse_i, mouseID in enumerate(mice):
			drive = data_dir + '/neural'
			mouse_dir = drive + '/' + mouseID + '/'
			if rel_to=='peak':
				learning_days = src.IO.get_carry_peak_learning_days(mouseID, m46_late)
			else:
				learning_days = src.IO.get_carry_thresh_learning_days(mouseID, m46_late)
			save_dir = mouse_dir + '/carry_analysis/learning/'
			fig_dir = save_dir + 'figures/'
			
			if learning_days==None:
				continue

			if calculate:
				reg_inds = src.utils.load_registered_cells(mouse_dir, learning_days)
				
				# grab the neural activity per day over all learning days
				act_spks = [[] for day in learning_days]
				dry_spks = [[] for day in learning_days]
				for i, day in enumerate(learning_days):
					print(mouseID, day)
					# load in the Cascade spikes
					s2p_fld = src.IO.get_s2p_fld(mouseID, day)
					
					spks = np.load(s2p_fld + 'cascade_spks.npy')
					# index by the registered cells
					reg_spks = spks[reg_inds[i], :]
					n_neurons=reg_spks.shape[0]

					# load in the behavioral times
					carry_times = np.load(s2p_fld + 'calcium_carry_times.npy')
					carry_labels = np.load(s2p_fld + 'calcium_carry_labels.npy')
					s_carry_times = carry_times[carry_labels==1]
					dry_carry_times = carry_times[carry_labels==2]

					print(len(s_carry_times), len(dry_carry_times))

					temp_a_spks = np.zeros((n_neurons, len(s_carry_times), time_in_trial))
					temp_d_spks = np.zeros((n_neurons, len(dry_carry_times), time_in_trial))
					for carry_i, carry in enumerate(s_carry_times.astype(int)):
						temp_a_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					for carry_i, carry in enumerate(dry_carry_times.astype(int)):
						temp_d_spks[:, carry_i, :] = reg_spks[:, (carry-t_pre):(carry+t_post)]
					act_spks[i]=temp_a_spks
					dry_spks[i]=temp_d_spks


				d_prime = np.zeros((n_neurons, len(learning_days)))
				for day_i, day in enumerate(learning_days):
					time_avg_a = np.nanmean(act_spks[day_i], axis=2)*30
					time_avg_d = np.nanmean(dry_spks[day_i], axis=2)*30

					trial_time_avg_a = np.nanmean(time_avg_a, axis=1)
					trial_time_avg_d = np.nanmean(time_avg_d, axis=1)
					trial_time_std_a = np.nanstd(time_avg_a, axis=1, ddof=1)
					trial_time_std_d = np.nanstd(time_avg_d, axis=1, ddof=1)
					d = d_prime_calc(trial_time_avg_a, trial_time_avg_d, trial_time_std_a, trial_time_std_d)
					
					d_prime[:, day_i] = d
					assert len(d)==n_neurons, 'took d prime over wrong axes'
					print('full pop: ', np.nanmean(d), '+-', stats.sem(d, nan_policy='omit'))
					print('range: ', np.nanmin(d), np.nanmax(d))
			
			
				if mouseID=='mouse46' and m46_late:
					np.save(save_dir+f'd_prime_late_days_rel_to_{rel_to}.npy', d_prime)
				else:
					np.save(save_dir+f'd_prime_diff_rel_to_{rel_to}.npy', d_prime)

			else:
				if mouseID=='mouse46' and m46_late:
					d_prime = np.load(save_dir+f'd_prime_late_days_rel_to_{rel_to}.npy')
				else:
					d_prime = np.load(save_dir+f'd_prime_diff_rel_to_{rel_to}.npy')
				n_neurons = len(d_prime)

			for day_i, day in enumerate(learning_days):
				# _, bins, _ = plt.hist(d_prime[:, pair_i], color='k')
				# plt.hist(d_prime[learning_decodable, pair_i], color='indianred', bins=bins)
				# plt.title(day_pair)
				# plt.show()
				print(np.sum(d_prime[:, day_i]>.3), np.sum(d_prime[:, day_i]>.3)/d_prime.shape[0])

			pop_d = np.nanmean(d_prime, axis=0)
			pop_sem = stats.sem(d_prime, axis=0, nan_policy='omit')

			plt.errorbar(learning_days, pop_d, yerr=pop_sem, color='k')
			plt.ylabel("Average d'")
			plt.xlabel('Training Day')
			plt.title(f'{mouseID} Active-Empty D-Prime')
			if mouseID=='mouse46' and m46_late:
				plt.savefig(fig_dir + f'Active-Empty D Prime Over Late Days rel to {rel_to}.png')
			else:
				plt.savefig(fig_dir + f'Active-Empty D Prime Over Days rel to {rel_to}.png')
			plt.close()

			if rel_to=='peak':
				day_of_interest = src.IO.get_carry_peak(mouseID, m46_late)
			else:
				day_of_interest = src.IO.get_carry_threshold_cross(mouseID, m46_late)

			rel_doi = int(np.where(np.array(learning_days, dtype=object)==day_of_interest)[0])
			print(rel_doi)
			rel_diff_mouse = np.zeros((n_neurons, 10))*np.nan
			rel_diff_mouse[:, (3-rel_doi):(3-rel_doi+5)] = d_prime
			rel_diff[mouse_i] = rel_diff_mouse


		rel_diff = np.concatenate(rel_diff, axis=0)
		pop_diff = np.nanmean(rel_diff, axis=0)
		pop_sem = stats.sem(rel_diff, axis=0, nan_policy='omit')
		
		fig, ax = plt.subplots()
		ax.plot([-3, -2, -1, 0, 1, 2], pop_diff[:6], color='darkred')
		ax.fill_between([-3, -2, -1, 0, 1, 2], pop_diff[:6]+pop_sem[:6], pop_diff[:6]-pop_sem[:6], color='darkred', alpha=.4)
		ax.axvline(0, color='k', ls='--')
		ax.set_ylabel("Mean d'")
		if rel_to=='peak':
			ax.set_xlabel('Training Day from Empty Grasp Peak')
		else:
			ax.set_xlabel('Training Day from Active-Empty Crossover')
		ax.set_title('Active-Empty Discriminability Over Learning')
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_xticks([-2, -1, 0, 1, 2])
		if m46_late:
			plt.savefig(multimouse_fig_dir + f'Single Day Active-Empty d prime Rel to {rel_to} m46 late days.png', dpi=660)
		else:
			plt.savefig(multimouse_fig_dir + f'Single Day Active-Empty d prime Rel to {rel_to}.png')
		plt.show()
	
		result = stats.ttest_rel(rel_diff[:, 1:3].flatten(), rel_diff[:, 4:6].flatten(), nan_policy='omit', alternative='less')
		print(result)