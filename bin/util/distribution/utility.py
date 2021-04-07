import pandas as pd
import numpy as np
import os

DIR_THIS_SCRIPT = os.path.dirname(os.path.realpath(__file__))

def get_ttc(state):
    """ Get the mean and standard deviation of Thermal Time Constant for the given state
    Args:
    - state: str, abbreviation name of states in US or provinces in Canada, example - 'CA'
    """
    ttc_all = pd.read_csv(os.path.join(DIR_THIS_SCRIPT,'RC&Teq/TTC.csv'), index_col=0)

    try:
        ttc_mean = ttc_all.loc[state, 'mean']
        ttc_sd = ttc_all.loc[state, 'sd']
        return ttc_mean, ttc_sd
    except:
        print("State/Province of {} not found".format(state))
        return None

def get_teq(state):
    """ Get the mean and standard deviation of equivalent temperature of other heat gains for the given state
    Args:
    - state: str, abbreviation name of states in US or provinces in Canada, example - 'CA'
    """
    teq_all = pd.read_csv(os.path.join(DIR_THIS_SCRIPT,'RC&Teq/Teq.csv'), index_col=0)

    try:
        teq_mean = teq_all.loc[state, 'mean']
        teq_sd = teq_all.loc[state, 'sd']
        return teq_mean, teq_sd
    except:
        print("State/Province of {} not found".format(state))
        return None

def get_comfort_temp(condition, method, ambTemp=20, hdd=5467, cdd=850):
    """ Get the mean and standard deviation of equivalent temperature of other heat gains for the given state
    Args:
    - condition: str, 'heating', 'cooling'
    - method: str, the method to determine the comfort temperature, currently four approaches are available
        -- 'Wang2020': Bayesian Inference from ASHRAE Database ,details refer to https://doi.org/10.1016/j.rser.2019.109593
        -- 'ASHRAE PMV': PMV-PPD model of ASHRAE Standard 55–2017
        -- 'ASHRAE adaptive': 90% acceptability based on Adapative Comfort Model of ASHRAE Standard 55–2017
            Linearly depends on 
        -- 'ResStock': Regressed from Residential Energy Consumption Survey, Page 41 https://www.nrel.gov/docs/fy18osti/68670.pdf
            Linearly depends on HDD18 and CDD18
    - ambTemp: float, Prevailing Mean Outdoor Temperature, which is arithmetic average of the mean daily outdoor temperatures over 
        no fewer than 7 and no more than 30 sequential days prior to the day in question, required by the adaptive comfort model,
        default 20 degC
    - hdd: Heating Degree Day (base temperature of 18), required by the ResStock method, default 5467 (medium level of US cities)
    - cdd: Cooling Degree Day (base temperature of 18), required by the ResStock method, default 850 (medium level of US cities)
    Return:
    - two tuples: (tsp_mean, tsp_sd), (trange_mean, trange_sd)
        -- tsp: temperature set point
        -- trange: acceptable comfort temperature range, temperature difference between the lower and upper bound
    """    
    assert condition in ['heating', 'cooling'], 'Condition of {} is not supported'.format(condition)
    assert method in ['Wang2020', 'ASHRAE PMV', 'ASHRAE adaptive', 'ResStock'], 'Method of {} is not supported'.format(method)

    if method == 'Wang2020':
        if condition == 'heating':
            return (23.6, 1.4), (1.2, 0.1)
        elif condition == 'cooling':
            return (22.7, 1.1), (1.2, 0.1)
    elif method == 'ASHRAE PMV':
        if condition == 'heating':
            return (22.3,   0), (3.7,   0)
        elif condition == 'cooling':
            return (25.4,   0), (2.9,   0)
    elif method == 'ASHRAE adaptive':
        heatSP = 0.325*ambTemp + 15.35
        coolSP = 0.31*ambTemp + 20.2
        return ((coolSP+heatSP)/2, 0), (coolSP-heatSP, 0)
    elif method == 'ResStock':
        heatSP = -0.0002*hdd + 20.97
        coolSP = 0.0006*cdd + 22.065
        return ((coolSP+heatSP)/2, 0), (coolSP-heatSP, 0)

