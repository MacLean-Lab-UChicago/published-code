'''
plot decodable cells piechart
'''
import matplotlib.pyplot as plt
import numpy as np
import src.utils
import src.IO

mice = ['mouse22', 'mouse25', 'mouse39', 'mouse35', 'mouse46', 'mouse51', 'mouse549']
multimouse_fig_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
plot_cells=False

colors = ['lightcoral', 'firebrick', 'darkred', 'darkgrey']
load_dir = '/media/elizawiener/e2176850-652a-4e33-9b85-5f3e649dbb1f/cross-mice active carry/'
mod_counts = src.IO.load_pickle(load_dir + 'mod_counts.pkl')
for m, mouseID in enumerate(mice):
	drive = src.IO.get_drive(mouseID)
	mouse_dir = drive + '/' + mouseID + '/'
	fig_dir = mouse_dir + f'carry_analysis/figures/'

	mouse_mod_counts = {key:0 for key in mod_counts.keys()}
	for key in mod_counts.keys():
		mouse_mod_counts[key] = mod_counts[key][m]

	plt.pie(list(mouse_mod_counts.values()), labels=list(mouse_mod_counts.keys()), autopct='%1.1f%%', colors=colors)
	plt.title(f'{mouseID} Cell Proportions')
	plt.savefig(fig_dir + 'mod cells pie chart.png')
	plt.show()

combined_mod_counts = {key:0 for key in mod_counts.keys()}
for key in mod_counts.keys():
	combined_mod_counts[key] = np.sum(mod_counts[key])
print(combined_mod_counts)
plt.pie(list(combined_mod_counts.values()), labels=list(combined_mod_counts.keys()), autopct='%1.1f%%', colors=colors)
plt.title(f'Cell Proportions')
plt.savefig(multimouse_fig_dir + 'mod cells pie chart.pdf', dpi=660)
plt.show()