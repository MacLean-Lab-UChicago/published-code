'''
logistic regression on drop modulated cells
requires:
	- drop modulated cell indices
	- decodable cell indices
	- excitatory/inhibitory cell indices
'''
import numpy as np
import src.IO
import src.utils

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
logit = True
permute = False
save_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/drops/'

all_cells_data = [[] for mouse in mice]
for mouse_i, mouseID in enumerate(mice):
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	carry_days = src.IO.get_carry_days(mouseID)

	# get NaN indices
	day = src.IO.get_carry_days(mouseID)[-1]
	s2p_fld = src.IO.get_s2p_fld(mouseID, day)
	NaN_cells = np.load(s2p_fld + 'NaN_containing_cells_bool.npy')

	# get red cell information
	# make sure to separate SST and PV
	red_cells = src.utils.load_red_cells(mouse_dir, carry_days)[~NaN_cells]

	# get decodability
	decodable = np.logical_or(np.load(s2p_fld + 'SVM_active_grasp_modulated_cells.npy'), np.load(s2p_fld + 'SVM_avgfr_active_grasp_modulated_cells.npy'))[~NaN_cells]

	# get drop modulation
	drop_mod = np.load(f'{mouse_dir}carry_analysis/drops/{mouseID}_all_days_drop_BH_mod_cells.npy')
	if not len(drop_mod)==(len(NaN_cells)-np.sum(NaN_cells)):
		drop_mod = drop_mod[~NaN_cells]

	n_neurons = len(red_cells)

	cell_data = np.zeros((n_neurons, 5), dtype=bool) # PV, SST, E, decodable, drop-modulated
	if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
		cell_data[:, 0] = red_cells
	else:
		cell_data[:, 1] = red_cells
	cell_data[:, 2] = ~red_cells
	cell_data[:, 3] = decodable
	cell_data[:, 4] = drop_mod

	all_cells_data[mouse_i] = cell_data

all_cells_data = np.concatenate(all_cells_data, axis=0)
n_neurons = all_cells_data.shape[0]
print(all_cells_data.shape)

if logit:
	from sklearn.metrics import accuracy_score
	import statsmodels.api as sm

	bias_term = np.ones(n_neurons)[:, np.newaxis]

	# logistic regression for I/E cell type to if drop modulated or not
	lr = sm.Logit(all_cells_data[:, 4], np.hstack((all_cells_data[:, :3], bias_term))).fit(disp=0)
	print(lr.summary())
	print(lr.pvalues)
	predicted = lr.predict(np.hstack((all_cells_data[:, :3], bias_term)))>.5
	a = accuracy_score(all_cells_data[:, 4], predicted)
	print(np.sum(predicted), a)

	# logistic regression for I/E type and decodable/not to drop-mod/not
	lr = sm.Logit(all_cells_data[:, 4], np.hstack((all_cells_data[:, :4], bias_term))).fit(disp=0)
	print(lr.summary())
	print(lr.pvalues)
	predicted = lr.predict(np.hstack((all_cells_data[:, :4], bias_term)))>.5
	a = accuracy_score(all_cells_data[:, 4], predicted)
	print(np.sum(predicted), a)

	for type, cell_quality in enumerate(['PV', 'SST', 'E', 'Decodable']):
		# logistic regression for single feature to drop-mod/not
		lr = sm.Logit(all_cells_data[:, 4], np.concatenate((all_cells_data[:, type, np.newaxis], bias_term), axis=1)).fit(disp=0)
		print(cell_quality)
		print(lr.summary())
		print(lr.pvalues)
		predicted = lr.predict(np.concatenate((all_cells_data[:, type, np.newaxis], bias_term), axis=1))>.5
		a = accuracy_score(all_cells_data[:, 4], predicted)
		print(np.sum(predicted), a)

if permute:
	import random
	import matplotlib.pyplot as plt
	from scipy.stats import percentileofscore
	n_resamples = 2500
	n_drop_mod = np.sum(all_cells_data[:, 4])
	resample_stats = np.zeros((n_resamples, 4)) # count for each resample of # of PV, SST, E, and decodable cells
	for resample in range(n_resamples):
		rand_mod = random.sample(range(n_neurons), n_drop_mod)
		resample_stats[resample, :] = np.sum(all_cells_data[rand_mod, :4], axis=0)

	for i, cell in enumerate(['PV', 'SST', 'E', 'Decodable']):
		plt.hist(resample_stats[:, i], color='grey')
		plt.axvline(np.sum(all_cells_data[all_cells_data[:, 4], i]), color='indianred')
		percentile = percentileofscore(resample_stats[:, i], np.sum(all_cells_data[all_cells_data[:, 4], i]))
		if percentile>50:
			p = 1-percentile/100
		else:
			p =  percentile/100
		plt.title(f'p={p}')
		plt.suptitle(cell)
		plt.savefig(save_dir + f'{cell} resampled vs drop cell counts.png')
		plt.show()