'''
the goal of this is to plot and get an idea for the kinematics of (different) carries
'''
import numpy as np
import functions
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import utils
import os

#mouseID = 'mouse46'
mice = ['mouse549']
#mice = ['mouse25']
bodyparts_for_centroid = ['d2knuckle', 'd3knuckle', 'wrist', 'wrist_outer']
bodyparts = ['d1middle', 'd2tip', 'd2middle', 'd2knuckle', 'd3tip', 'd3middle', 'd3knuckle', 
'd4tip', 'd4middle', 'wrist', 'wrist_outer', 'elbow', 'elbow_crook', 'pedestal']
t_pre = 25 # in behavioral frames, before rotation
t_post = 25 # behavior frames, after rotation
beh_duration=t_pre+t_post

plot_trajectories=True
plot_normal=False
plot_splay=False
plot_aperature=False

for mouseID in mice:
	cohort = utils.get_cohort(mouseID)
	days = utils.get_carry_days(mouseID)
	drive = utils.get_drive(mouseID)

	dlc_datapath = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/kinematics_3d/' + cohort + '/' + mouseID + '/data/'
	mouse_datapath = drive + '/' + mouseID + '/'

	pkl_savepath = mouse_datapath + 'cam_carry_alignment/'
	figure_savepath = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/kinematics_3d/' + cohort + '/' + mouseID + '/figures/active_grasp/'

	if not os.path.isdir(figure_savepath):
		os.mkdir(figure_savepath)

	normal = [[] for i in days]
	aperature = [[] for i in days]
	splay = [[] for i in days]
	dlc_data = [[] for i in days]
	paw_centroid = [[] for i in days]
	labels_multiday = []

	for i, day in enumerate(days):
		pkl_filepath = pkl_savepath + f'{mouseID}_{day}_carries_aligned.pkl'
		carry_times = utils.load_pickle(pkl_filepath)
		normal[i] = [[] for key in carry_times.keys() if len(carry_times[key]['labels'])>0]
		aperature[i] = [[] for key in carry_times.keys() if len(carry_times[key]['labels'])>0]
		splay[i] = [[] for key in carry_times.keys() if len(carry_times[key]['labels'])>0]
		dlc_data[i] = [[] for key in carry_times.keys() if len(carry_times[key]['labels'])>0]
		paw_centroid[i] = [[] for key in carry_times.keys() if len(carry_times[key]['labels'])>0]
		j=0
		for key in carry_times.keys():
			if len(carry_times[key]['labels'])==0:
				continue
			else:
				labels = carry_times[key]['labels']
				rotations = carry_times[key]['rotation_time'][labels!=3].astype(int)
				labels = labels[labels!=3]
				labels_multiday.extend(labels)

				df = utils.load_dlc(dlc_datapath, mouseID, day, key)
				orientation, normal[i][j] = functions.get_orientation_less_NaNs(df, rotations-t_pre, frame_length=beh_duration)
				aperature[i][j], avg_aperature = functions.get_aperature(df, rotations-t_pre, frame_length=beh_duration)
				splay[i][j], avg_splay = functions.get_splay(df, rotations-t_pre, frame_length=beh_duration)

				dlc_data[i][j] = np.zeros((len(labels), len(bodyparts)*3, beh_duration))
				paw_centroid[i][j] = np.zeros((len(labels), 3, beh_duration))
				paw_x, paw_y, paw_z = functions.get_paw_centroid_with_NaNs(df, bodyparts_for_centroid)
				for carry_i, rotation in enumerate(rotations):
					start=rotation-t_pre
					end=rotation+t_post
					paw_centroid[i][j][carry_i, 0, :], paw_centroid[i][j][carry_i, 1, :], paw_centroid[i][j][carry_i, 2, :] = paw_x[start:end], paw_y[start:end], paw_z[start:end]
					for body_i, bodypart in enumerate(bodyparts):
						coord_idx = 0
						for coordinate in ['x', 'y', 'z']:
							dlc_data[i][j][carry_i, body_i*3+coord_idx, :]=np.array(df['etw'][bodypart][coordinate][start:end])
							coord_idx+=1
				j+=1
		normal[i] = np.concatenate(normal[i], axis=0)
		aperature[i] = np.concatenate(aperature[i], axis=0)
		splay[i] = np.concatenate(splay[i], axis=0)
		dlc_data[i] = np.concatenate(dlc_data[i], axis=0)
		paw_centroid[i] = np.concatenate(paw_centroid[i], axis=0)

	print(len(labels_multiday))

	normal = np.concatenate(normal, axis=0)
	aperature = np.concatenate(aperature, axis=0)
	splay = np.concatenate(splay, axis=0)
	dlc_data = np.concatenate(dlc_data, axis=0)
	paw_centroid = np.concatenate(paw_centroid, axis=0)

	print(normal.shape)
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

	if plot_trajectories:
		# fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'})
		# for carry_type in np.unique(labels_multiday).astype(int):
		# 	plotting_data=dlc_data[labels_multiday==carry_type, :, :]
		# 	avg = np.nanmean(plotting_data, axis=0)
		# 	for bp_idx, bp in enumerate(bodyparts[:-1]):
		# 		ax[carry_type].plot3D(avg[bp_idx*3, :], -avg[bp_idx*3+1, :], -avg[bp_idx*3+2, :], label=bp)
		# ax[1].legend(loc='center', bbox_to_anchor=(0, -.25), ncol=len(bodyparts)/2)	
		# ax[0].set_title('Drop')
		# ax[1].set_title('Success')
		# ax[2].set_title('Dry')
		# fig.suptitle(f'{mouseID} Averaged Carry Trajectories')
		# plt.show()

		bkg_color = (0, 0, 0, 0.06)
		fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
		for carry_type in np.sort(np.unique(labels_multiday).astype(int))[::-1]:
			if carry_type==0:
				continue
			elif carry_type==1:
				label='active grasp'
				color='indianred'
				alpha=0.07
			elif carry_type==2:
				label='empty grasp'
				color='k'
				alpha=0.03
			for counter, carry in enumerate(np.where(labels_multiday==carry_type)[0]):
				ax.plot3D(paw_centroid[carry, 0, :], -paw_centroid[carry, 1, :], -paw_centroid[carry, 2, :], color=color, alpha=alpha)
		for carry_type in np.unique(labels_multiday).astype(int):
			if carry_type==0:
				continue
			elif carry_type==1:
				label='active grasp'
				color='indianred'
			elif carry_type==2:
				label='empty grasp'
				color='k'
			mean_carry = np.nanmean(paw_centroid[labels_multiday==carry_type, :, :], axis=0)
			ax.plot3D(mean_carry[0, :], -mean_carry[1, :], -mean_carry[2, :], color=color, label=label, lw=2)


		# ax.plot3D(ped_x, -ped_y, -ped_z, '*', color='k', label='pedestal')
		ax.view_init(35, -146)

		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])

		# Remove axis lines
		ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

		# change the background color
		ax.xaxis.set_pane_color(bkg_color) # White (RGBA: 1 for white, 1 for opaque)
		ax.yaxis.set_pane_color(bkg_color)
		ax.zaxis.set_pane_color(bkg_color)

		x_min=23.3
		y_min=-8
		z_min=-57.5
		largest_gap = 8.5

		ax.set_xlim((x_min, x_min+largest_gap))
		ax.set_ylim((y_min, y_min+largest_gap))
		ax.set_zlim((z_min, z_min+largest_gap))

		#ax.legend()
		#ax.set_title(f'{mouseID.capitalize()} Carry Trajectories')
		fig.savefig(figure_savepath + 'paw centroid trajectory.png', dpi=660)
		plt.show()

		bkg_color = (0, 0, 0, 0.06)
		fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
		alpha=0.03
		for carry in range(paw_centroid.shape[0]):
			ax.plot3D(paw_centroid[carry, 0, :], -paw_centroid[carry, 1, :], -paw_centroid[carry, 2, :], color='k', alpha=alpha)
		mean_carry = np.nanmean(paw_centroid[:, :, :], axis=0)
		ax.plot3D(mean_carry[0, :], -mean_carry[1, :], -mean_carry[2, :], color='r', lw=2)


		# ax.plot3D(ped_x, -ped_y, -ped_z, '*', color='k', label='pedestal')
		ax.view_init(35, -146)

		# Hide grid lines
		ax.grid(False)

		# Hide axes ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])

		# Remove axis lines
		ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
		ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

		# change the background color
		ax.xaxis.set_pane_color(bkg_color) # White (RGBA: 1 for white, 1 for opaque)
		ax.yaxis.set_pane_color(bkg_color)
		ax.zaxis.set_pane_color(bkg_color)

		#ax.set_title(f'{mouseID.capitalize()} Carry Trajectories')
		fig.savefig(figure_savepath + 'paw centroid trajectory all carries.pdf', dpi=660)
		plt.show()


	if plot_normal:
		figx, axx = plt.subplots(3, 1)
		figy, axy = plt.subplots(3, 1)
		figz, axz = plt.subplots(3, 1)
		for carry_type in np.unique(labels_multiday).astype(int):
			norm = normal[labels_multiday==carry_type, :, :]
			for carry in range(norm.shape[0]):
				axx[carry_type].plot(norm[carry, 0, :], color='k', alpha=0.5)
				axy[carry_type].plot(norm[carry, 1, :], color='k', alpha=0.5)
				axz[carry_type].plot(norm[carry, 2, :], color='k', alpha=0.5)
			avg_norm = np.nanmean(norm, axis=0)
			axx[carry_type].plot(avg_norm[0, :], color='r')
			axy[carry_type].plot(avg_norm[1, :], color='r')
			axz[carry_type].plot(avg_norm[2, :], color='r')
		axx[0].set_title('Drop')
		axx[1].set_title('Success')
		axx[2].set_title('Dry')
		axy[0].set_title('Drop')
		axy[1].set_title('Success')
		axy[2].set_title('Dry')
		axz[0].set_title('Drop')
		axz[1].set_title('Success')
		axz[2].set_title('Dry')
		figx.suptitle(f'{mouseID} Palm Norm X')
		figy.suptitle(f'{mouseID} Palm Norm Y')
		figz.suptitle(f'{mouseID} Palm Norm Z')
		figx.savefig(figure_savepath + f'{mouseID} Palm Norm X.png')
		figy.savefig(figure_savepath + f'{mouseID} Palm Norm Y.png')
		figz.savefig(figure_savepath + f'{mouseID} Palm Norm Z.png')
		plt.show()

	if plot_aperature:
		figx, axx = plt.subplots(3, 1)
		figy, axy = plt.subplots(3, 1)
		figz, axz = plt.subplots(3, 1)
		figavg, axavg = plt.subplots(3, 1)
		for carry_type in np.unique(labels_multiday).astype(int):
			ap = aperature[labels_multiday==carry_type, :, :]
			aperature_avg = np.nanmean(ap, axis=1)
			for carry in range(ap.shape[0]):
				axx[carry_type].plot(ap[carry, 0, :], color='k', alpha=0.5)
				axy[carry_type].plot(ap[carry, 1, :], color='k', alpha=0.5)
				axz[carry_type].plot(ap[carry, 2, :], color='k', alpha=0.5)
				axavg[carry_type].plot(aperature_avg[carry, :], color='k', alpha=0.5)
			avg_ap = np.nanmean(ap, axis=0)
			avg_avg_ap = np.nanmean(aperature_avg, axis=0)
			axx[carry_type].plot(avg_ap[0, :], color='r')
			axy[carry_type].plot(avg_ap[1, :], color='r')
			axz[carry_type].plot(avg_ap[2, :], color='r')
			axavg[carry_type].plot(avg_avg_ap, color='r')
		axx[0].set_title('Drop')
		axx[1].set_title('Success')
		axx[2].set_title('Dry')
		axy[0].set_title('Drop')
		axy[1].set_title('Success')
		axy[2].set_title('Dry')
		axz[0].set_title('Drop')
		axz[1].set_title('Success')
		axz[2].set_title('Dry')
		axavg[0].set_title('Drop')
		axavg[1].set_title('Success')
		axavg[2].set_title('Dry')
		figx.suptitle(f'{mouseID} d2-wrist aperature')
		figy.suptitle(f'{mouseID} d3-wrist aperature')
		figz.suptitle(f'{mouseID} d4-wrist aperature')
		figavg.suptitle(f'{mouseID} Average All Digits Aperature')
		figx.savefig(figure_savepath + f'{mouseID} d2-wrist aperature.png')
		figy.savefig(figure_savepath + f'{mouseID} d3-wrist aperature.png')
		figz.savefig(figure_savepath + f'{mouseID} d4-wrist aperature.png')
		figavg.savefig(figure_savepath + f'{mouseID} Average All Digits Aperature.png')

		plt.show()
		
	if plot_splay:
		figx, axx = plt.subplots(3, 1)
		figy, axy = plt.subplots(3, 1)
		figz, axz = plt.subplots(3, 1)
		figavg, axavg = plt.subplots(3, 1)
		for carry_type in np.unique(labels_multiday).astype(int):
			spread = splay[labels_multiday==carry_type, :, :]
			spread_avg = np.nanmean(spread, axis=1)
			for carry in range(spread.shape[0]):
				axx[carry_type].plot(spread[carry, 0, :], color='k', alpha=0.5)
				axy[carry_type].plot(spread[carry, 1, :], color='k', alpha=0.5)
				axz[carry_type].plot(spread[carry, 2, :], color='k', alpha=0.5)
				axavg[carry_type].plot(spread_avg[carry, :], color='k', alpha=0.5)
			avg_spread = np.nanmean(spread, axis=0)
			avg_avg_spread = np.nanmean(spread_avg, axis=0)
			axx[carry_type].plot(avg_spread[0, :], color='r')
			axy[carry_type].plot(avg_spread[1, :], color='r')
			axz[carry_type].plot(avg_spread[2, :], color='r')
			axavg[carry_type].plot(avg_avg_spread, color='r')
		axx[0].set_title('Drop')
		axx[1].set_title('Success')
		axx[2].set_title('Dry')
		axy[0].set_title('Drop')
		axy[1].set_title('Success')
		axy[2].set_title('Dry')
		axz[0].set_title('Drop')
		axz[1].set_title('Success')
		axz[2].set_title('Dry')
		axavg[0].set_title('Drop')
		axavg[1].set_title('Success')
		axavg[2].set_title('Dry')
		figx.suptitle(f'{mouseID} d1-d2 splay')
		figy.suptitle(f'{mouseID} d2-d3 splay')
		figz.suptitle(f'{mouseID} d3-d4 splay')
		figavg.suptitle(f'{mouseID} Average All Digits Splay')
		figx.savefig(figure_savepath + f'{mouseID} d1-d2 splay.png')
		figy.savefig(figure_savepath + f'{mouseID} d2-d3 splay.png')
		figz.savefig(figure_savepath + f'{mouseID} d3-d4 splay.png')
		figavg.savefig(figure_savepath + f'{mouseID} Average All Digits Splay.png')
		plt.show()
		

