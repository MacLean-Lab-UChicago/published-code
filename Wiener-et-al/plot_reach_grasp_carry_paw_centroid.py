'''
plot trajectories of reach, carry, and grasp paw centroid
'''
import numpy as np
import functions
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import utils
import os
import glob
from pathlib import Path

data_dir = Path.cwd() / 'data/kinematics/'

mice = ['mouse25']
bodyparts_for_centroid = ['d2knuckle', 'd3knuckle', 'wrist', 'wrist_outer']
bodyparts = ['d1middle', 'd2tip', 'd2middle', 'd2knuckle', 'd3tip', 'd3middle', 'd3knuckle', 
'd4tip', 'd4middle', 'wrist', 'wrist_outer', 'elbow', 'elbow_crook', 'pedestal']
t_pre = 0 # in behavioral frames, before start
t_post = 50 # behavior frames, after start
beh_duration=t_pre+t_post
behaviors = ['reach', 'grasp', 'carry']

# aesthetic choices
bkg_color = (0, 0, 0, 0.06)
view1, view2 = 35, -146
alpha=0.1

for mouseID in mice:
	day = sorted(glob.glob(f'{data_dir}/{mouseID}/*/*_reach_times.pkl'))[-1].split('/')[-2]
	dlc_datapath = f'{data_dir}/{mouseID}/dlc/'
	pkl_savepath = f'{data_dir}/{mouseID}/{day}/'
	figure_savepath =  f'{Path.cwd() / 'data'}/results/single_mouse_results/{mouseID}/'

	pkl_filepath = pkl_savepath + f'{mouseID}_{day}_reach_times.pkl'
	event_times = utils.load_pickle(pkl_filepath)

	try:
		dlc_data = [[] for key in event_times.keys() if len(event_times[key]['labels'])>0]
		paw_centroid = [[] for key in event_times.keys() if len(event_times[key]['labels'])>0]
	except:
		no_labels=True
		i=2
		while no_labels:
			day = sorted(glob.glob(f'{data_dir}/{mouseID}/*/*_reach_times.pkl'))[-1].split('/')[i]
			print(day)

			dlc_datapath = f'{data_dir}/{mouseID}/dlc/'

			pkl_savepath = f'{data_dir}/{mouseID}/{day}/'
			figure_savepath =  f'{data_dir}/{mouseID}/figures/'

			pkl_filepath = pkl_savepath + f'{mouseID}_{day}_reach_times.pkl'
			event_times = utils.load_pickle(pkl_filepath)

			try:
				dlc_data = [[] for key in event_times.keys() if len(event_times[key]['labels'])>0]
				paw_centroid = [[] for key in event_times.keys() if len(event_times[key]['labels'])>0]
				no_labels=False
			except:
				i+=1
	j=0
	day_labels = []
	pellet_x, pellet_y, pellet_z = [], [], []
	for key in event_times.keys():
		if len(event_times[key]['labels'])==0:
			continue
		else:
			labels = event_times[key]['labels']
			starts = event_times[key]['starts'][np.logical_or(np.logical_or(labels==0, labels==1), labels==2)].astype(int)
			labels = labels[np.logical_or(np.logical_or(labels==0, labels==1), labels==2)]
			day_labels.extend(labels)

			df = utils.load_dlc(dlc_datapath, mouseID, day, key)
			dlc_data[j] = np.zeros((len(labels), len(bodyparts)*3, beh_duration))
			paw_centroid[j] = np.zeros((len(labels), 3, beh_duration))
			paw_x, paw_y, paw_z = functions.get_paw_centroid_with_NaNs(df, bodyparts_for_centroid)
			pel_x, pel_y, pel_z = functions.get_x_y_z(df, 'pellet')
			pellet_x.extend(pel_x)
			pellet_y.extend(pel_y)
			pellet_z.extend(pel_z)
			for event_i, start in enumerate(starts):
				start_idx=start-t_pre
				end_idx=start+t_post
				paw_centroid[j][event_i, 0, :], paw_centroid[j][event_i, 1, :], paw_centroid[j][event_i, 2, :] = paw_x[start_idx:end_idx], paw_y[start_idx:end_idx], paw_z[start_idx:end_idx]
				for body_i, bodypart in enumerate(bodyparts):
					coord_idx = 0
					for coordinate in ['x', 'y', 'z']:
						dlc_data[j][event_i, body_i*3+coord_idx, :]=np.array(df['etw'][bodypart][coordinate][start_idx:end_idx])
						coord_idx+=1
			j+=1
	dlc_data = np.concatenate(dlc_data, axis=0)
	paw_centroid = np.concatenate(paw_centroid, axis=0)

	labels = np.array(day_labels)

	print(dlc_data.shape)
	print(paw_centroid.shape)

	ped_x = dlc_data[:, -3, :]
	ped_y = dlc_data[:, -2, :]
	ped_z = dlc_data[:, -1, :]
	if not np.nanmedian(ped_x) > 0 and not np.nanmedian(ped_y) > 0:
		ped_x=np.nanmean(ped_x)
		ped_y=np.nanmean(ped_y)
		ped_z=np.nanmean(ped_z)
	else:
		ped_x=np.nanmedian(ped_x)
		ped_y=np.nanmedian(ped_y)
		ped_z=np.nanmedian(ped_z)
	dlc_data=dlc_data[:, :-3, :]
	
	if not np.nanmedian(pellet_x) > 0 and not np.nanmedian(pellet_y) > 0:
		pellet_x=np.nanmean(pellet_x)
		pellet_y=np.nanmean(pellet_y)
		pellet_z=np.nanmean(pellet_z)
	else:
		pellet_x=np.nanmedian(pellet_x)
		pellet_y=np.nanmedian(pellet_y)
		pellet_z=np.nanmedian(pellet_z)

	print(np.nanmin(paw_centroid[:, 0, :]), np.nanmin(-paw_centroid[:, 1, :]), np.nanmin(-paw_centroid[:, 2, :]))
	print(np.nanmax(paw_centroid[:, 0, :]), np.nanmax(-paw_centroid[:, 1, :]), np.nanmax(-paw_centroid[:, 2, :]))
	x_min, y_min, z_min = np.nanmin(paw_centroid[:, 0, :]), np.nanmin(-paw_centroid[:, 1, :]), np.nanmin(-paw_centroid[:, 2, :])
	x_max, y_max, z_max = np.nanmax(paw_centroid[:, 0, :]), np.nanmax(-paw_centroid[:, 1, :]), np.nanmax(-paw_centroid[:, 2, :])
	largest_gap = np.max([x_max-x_min, y_max-y_min, z_max-z_min])
	
	for i, label in enumerate(np.unique(labels)):
		fig, ax = plt.subplots(figsize=(10, 7.5), subplot_kw={'projection': '3d'})
		ax.plot3D(ped_x, -ped_y, -ped_z, 'ob', ms=5, label='pedestal')
		ax.plot3D(pellet_x, -pellet_y, -pellet_z, 'og', ms=10, label='pellet')
		for event in range(paw_centroid.shape[0]):
			if labels[event]==label:
				ax.plot3D(paw_centroid[event, 0, :], -paw_centroid[event, 1, :], -paw_centroid[event, 2, :], color='k', alpha=alpha)
		mean_event = np.nanmean(paw_centroid[labels==label, :, :], axis=0)
		ax.plot3D(mean_event[0, :], -mean_event[1, :], -mean_event[2, :], color='indianred', lw=2)
		


		ax.view_init(view1, view2)

		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])

		ax.set_xlim((x_min, x_min+largest_gap))
		ax.set_ylim((y_min, y_min+largest_gap))
		ax.set_zlim((z_min, z_min+largest_gap))

		ax.plot3D([x_min, x_min+2.5], [y_min+largest_gap, y_min+largest_gap], [z_min, z_min], 'k')
		ax.plot3D([x_min, x_min], [y_min+largest_gap, y_min+largest_gap-2.5], [z_min, z_min], 'k')
		ax.plot3D([x_min, x_min], [y_min+largest_gap, y_min+largest_gap], [z_min, z_min+2.5], 'k')

		# Remove axis lines
		ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

		# change the background color
		ax.xaxis.set_pane_color(bkg_color) # White (RGBA: 1 for white, 1 for opaque)
		ax.yaxis.set_pane_color(bkg_color)
		ax.zaxis.set_pane_color(bkg_color)

		ax.set_title(f'{behaviors[i]}, n={np.sum(labels==label)}')
		fig.savefig(figure_savepath + f'paw centroid trajectory all {behaviors[i]} {day}.pdf', dpi=450)
		plt.show()