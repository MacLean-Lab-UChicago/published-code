import numpy as np
import pickle
import glob
#import pandas as pd
import src.utils
from pathlib import Path

def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(filename,var):
    with open(filename, "wb") as file:
        pickle.dump(var, file)

def get_carry_days(mouseID):
    if mouseID=='mouse22':
        days = ['042524', '042824', '042924', '043024', '050124']
    elif mouseID=='mouse25':
        days = ['042224', '042324', '042424', '042524']
    elif mouseID=='mouse39':
        days = ['042324', '042424', '042524', '042824']
    elif mouseID=='mouse35':
        days = ['070724', '070824', '070924', '071024'] 
    elif mouseID=='mouse549':
        days = ['081624', '081724', '081824', '081924']
    elif mouseID=='mouse51':
        days = ['081824', '081924', '082024', '082124']
    elif mouseID=='mouse46':
        days = ['081924', '082024', '082124', '082224']
    else:
        days=None
    return days

def get_carry_learning_days(mouseID):
    if mouseID=='mouse22':
        days = ['042524', '042824', '042924', '043024', '050124'] # maybe just do that?
        #days=None
    elif mouseID=='mouse25':
        days = ['042224', '042324', '042424', '042524', '042824']
    elif mouseID=='mouse39':
        days = ['042224', '042324', '042424', '042524', '042824']
    elif mouseID=='mouse35':
        days = ['070724', '070824', '070924', '071024', '071124'] 
    elif mouseID=='mouse549':
        days = ['081624', '081724', '081824', '081924', '082024']
    elif mouseID=='mouse51':
        days = ['081824', '081924', '082024', '082124', '082224']
    elif mouseID=='mouse46':
        #days=None
        days = ['081824', '081924', '082024', '082124', '082224'] # need to decide between the 18th and the 21st
    else:
        days=None
    return days

def get_carry_peak_learning_days(mouseID, m46_late):
    if mouseID=='mouse22':
        days = ['042524', '042824', '042924', '043024', '050124'] # maybe just do that?
        #days=None
    elif mouseID=='mouse25':
        days = ['042124', '042224', '042324', '042424', '042524']
    elif mouseID=='mouse39':
        days = ['042124', '042224', '042324', '042424', '042524']
    elif mouseID=='mouse35':
        days = ['070524', '070624', '070724', '070824', '070924'] 
    elif mouseID=='mouse46':
        if m46_late:
            days = ['082224', '082324', '082424', '082524', '082624']
        else:
            days = ['081724', '081824', '081924', '082024', '082124']
    elif mouseID=='mouse51':
        days = ['081724', '081824', '081924', '082024', '082124'] 
    elif mouseID=='mouse549':
        days = ['081424', '081524', '081624', '081724', '081824']
    else:
        days=None
    return days

def get_carry_thresh_learning_days(mouseID, m46_late):
    if mouseID=='mouse22':
        days = ['042524', '042824', '042924', '043024', '050124']
    elif mouseID=='mouse25':
        days = ['042324', '042424', '042524', '042824', '042924']
    elif mouseID=='mouse39':
        days = ['042224', '042324', '042424', '042524', '042824']
    elif mouseID=='mouse35':
        days = ['070924', '071024', '071124', '071224', '071324'] 
    elif mouseID=='mouse46':
        if m46_late:
            days = ['082324', '082424', '082524', '082624', '082724']
        else:
            days = ['081924', '082024', '082124', '082224', '082324']
    elif mouseID=='mouse51':
        days = ['081924', '082024', '082124', '082224', '082324'] 
    elif mouseID=='mouse549':
        days = ['081824', '081924', '082024', '082124', '082224']
    else:
        days=None
    return days

def get_carry_peak(mouseID, m46_late):
    if mouseID=='mouse22':
        peak = ['042524']
    elif mouseID=='mouse25':
        peak = ['042224']
    elif mouseID=='mouse39':
        peak = ['042324']
    elif mouseID=='mouse35':
        peak = ['070724'] 
    elif mouseID=='mouse549':
        peak = ['081624']
    elif mouseID=='mouse51':
        peak = ['081924']
    elif mouseID=='mouse46':
        if m46_late:
            peak = ['082424']
        else:
            peak = ['081924']
    else:
        peak=None
    return peak

def get_carry_crossover(mouseID, m46_late):
    if mouseID=='mouse22':
        peak = ['042824']
    elif mouseID=='mouse25':
        peak = ['042524']
    elif mouseID=='mouse39':
        peak = ['042424']
    elif mouseID=='mouse35':
        peak = ['071124'] 
    elif mouseID=='mouse549':
        peak = ['082024']
    elif mouseID=='mouse51':
        peak = ['082124']
    elif mouseID=='mouse46':
        if m46_late:
            peak = ['082524']
        else:
            peak = ['082124']
    else:
        peak=None
    return peak

def get_days(mouseID):
    data_dir = Path.cwd() / 'data'
    day_paths = sorted(glob.glob(data_dir+'/neural/'+mouseID+'/*24'))
    days = [day.split('/')[-1] for day in day_paths]
    return days

def get_first_day(mouseID):
    data_dir = Path.cwd() / 'data'
    day_paths = sorted(glob.glob(data_dir+'/neural/'+mouseID+'/*24'))
    days = [day.split('/')[-1] for day in day_paths]
    if (mouseID=='mouse22')|(mouseID=='mouse25')|(mouseID=='mouse39'):
        return days[0]
    else:
        for day in days:
            s2p_fld = get_s2p_fld(mouseID, day)
            try:
                np.load(s2p_fld + 'event_labels.npy')
                return day
            except:
                not_yet=True


def get_s2p_fld(mouseID, day):
    data_dir = Path.cwd() / 'data'
    mouse_dir = data_dir + '/neural/' + mouseID + '/'
    calcium_data_path = mouse_dir + day
    s2p_fld = calcium_data_path + '/'
    return s2p_fld

def load_cam_event_times(mouseID,day):
    s2p_fld = get_s2p_fld(mouseID,day)
    cam_time_fn = f"{s2p_fld}/cam_event_times.pkl"
    event_times = load_pickle(cam_time_fn)
    #get rid of None entries
    filt_event_times = {k: v for k, v in event_times.items() if v is not None}
    return filt_event_times

def load_event_labels(mouseID,day):
    s2p_fld = get_s2p_fld(mouseID,day)
    return np.load(f"{s2p_fld}/event_labels.npy")

def load_spks_and_events(mouseID,day):
    s2p_fld = get_s2p_fld(mouseID,day)
    spks = np.load(f"{s2p_fld}/cascade_spks.npy")
    event_times = np.load(f"{s2p_fld}/calcium_event_times.npy")
    event_labels = np.load(f"{s2p_fld}/event_labels.npy")
    return spks, event_times,event_labels

def load_hdf(file):
    df = pd.read_hdf(file)
    return df

def get_spks_for_beh(mouseID,day,beh,t_pre,t_post):#(spks,beh_times,t_pre,t_post):
    '''spks with shape: n_neurons x n_timepts x n_trials'''
    s2p_fld = get_s2p_fld(mouseID,day)
    spks = np.load(s2p_fld+'/cascade_spks.npy')
    event_times = np.load(s2p_fld + 'calcium_event_times.npy')
    event_labels = np.load(s2p_fld + 'event_labels.npy').astype(int)
    beh_labels = src.utils.get_labels_for_behavior(event_labels,beh)
    beh_times = event_times[beh_labels]
    beh_trials = np.zeros((spks.shape[0],t_pre+t_post,beh_times.shape[0]))
    trs_to_remove = []
    for i,t in enumerate(beh_times.astype(int)):
        if (t+t_post < (spks.shape[1]-32)) and (t-t_pre > 32):
            beh_trials[:,:,i] = spks[:,(t-t_pre):(t+t_post)]
        else:
            trs_to_remove+=[i]
    #flag trials before or after cascade edge effects - set to nan
    #first check that there aren't nans in beh_trials for other reasons
    #assert not np.any(np.isnan(beh_trials)), 'nans in spks'
    if len(trs_to_remove)>=1:
        print(f'setting {trs_to_remove} to nan because of cascade edge effects')
        beh_trials[:,:,trs_to_remove] = np.nan
    return beh_trials
