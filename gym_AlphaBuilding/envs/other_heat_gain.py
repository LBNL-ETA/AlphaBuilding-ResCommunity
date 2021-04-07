import numpy as np

# Schedule from DOE Reference Model
OCC_SCHD_DOE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 0.9, 0.4, 0.2, 0.2, 0.2,
                     0.2, 0.2, 0.2, 0.2, 0.3, 0.6,
                     0.9, 0.9, 0.9, 1.0, 1.0, 1.0])
LIG_SCHD = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.4,
                     0.4, 0.4, 0.2, 0.1, 0.1, 0.1,
                     0.1, 0.1, 0.1, 0.2, 0.4, 0.6,
                     0.8, 1.0, 1.0, 0.7, 0.4, 0.2])
MIS_SCHD_DOE = np.array([0.6, 0.6, 0.6, 0.5, 0.5, 0.6,
                     0.7, 0.7, 0.6, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.6, 0.6, 0.7, 0.9,
                     0.9, 1.0, 1.0, 1.0, 0.8, 0.7])
OCC_SCHD_ECO_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 0.9, 0.4, 0.2, 0.2, 0.2,
                     0.2, 0.2, 0.2, 0.2, 0.3, 0.6,
                     0.9, 0.9, 0.9, 1.0, 1.0, 1.0])
OCC_SCHD_ECO_2 = np.ones(24)*0.9
MIS_SCHD_ECO_1 = np.array([0.6, 0.6, 0.6, 0.5, 0.5, 0.6,
                     0.7, 0.7, 0.6, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.6, 0.6, 0.7, 0.9,
                     0.9, 1.0, 1.0, 1.0, 0.8, 0.7])
MIS_SCHD_ECO_2 = np.ones(24)*0.9

SOL_SCHD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.1, 0.4, 0.6, 0.9,
                     1.0, 1.0, 0.9, 0.7, 0.4, 0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

OCC_DEN = 2         # unit W/m2, 250m2 for 5 people, 100W each person
LIG_DEN = 3.1       # unit W/m2, from DOE Reference Model: 2.47(Hardwired)+0.62(Plug-in)
MIS_DEN = 14        # unit W/m2, 250m2 consumes 3500W

def get_intHG_schd(method, sample_size, day_of_week):
    '''Get the internal heat gain (occupant, lighting, miscellaneous load) schedule
    Args:
    - method: str, method used to get internal heat gain load
        'DOE': use DOE Reference model
        'Ecobee': use the schedule inferred from Ecobee dataset
    - sample_size: int, number of households
    - day_of_week: int, Monday=0, Sunday=6, same as pandas  
    '''
    assert method in ['DOE', 'Ecobee'], 'Method of {} is currently not supported'
    profile_all = np.empty((sample_size,24))

    if method == 'DOE':
        profile = OCC_SCHD_DOE*OCC_DEN+LIG_SCHD*LIG_DEN+MIS_SCHD_DOE*MIS_DEN
        profile_all[:,:] = profile_normalization(profile)
    elif method == 'Ecobee':
        cluster_ratio_weekday = [0.8, 0.2]
        cluster_ratio_weekend = [0.3, 0.7]
        profile_1 = OCC_SCHD_ECO_1*OCC_DEN+LIG_SCHD*LIG_DEN+MIS_SCHD_ECO_1*MIS_DEN
        profile_2 = OCC_SCHD_ECO_2*OCC_DEN+LIG_SCHD*LIG_DEN+MIS_SCHD_ECO_2*MIS_DEN
        profile_1 = profile_normalization(profile_1)
        profile_2 = profile_normalization(profile_2)
        if day_of_week < 5:
            cluster_ratio = np.array(cluster_ratio_weekday)  # np array can use method of cumsum
        else:
            cluster_ratio = np.array(cluster_ratio_weekend)
        
        sample = np.random.uniform(0,1,sample_size)

        profile_all[sample<=cluster_ratio.cumsum()[0],:] = profile_1
        profile_all[(cluster_ratio.cumsum()[0]<sample) & \
            (sample<=cluster_ratio.cumsum()[1]),:] = profile_2

    return profile_all

def profile_normalization(profile):
    peak = profile.max()
    profile_normalized = profile/peak
    return profile_normalized

def get_solarHG_schd():
    '''Get the solar heat gain schedule
    Args:
    -   
    '''
    return SOL_SCHD
