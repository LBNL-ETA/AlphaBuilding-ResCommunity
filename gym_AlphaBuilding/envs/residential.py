import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_AlphaBuilding.envs.other_heat_gain import get_intHG_schd, get_solarHG_schd

import pandas as pd
import numpy as np
from datetime import timedelta

class AlphaResEnv(gym.Env):
    """ AlphaResEnv is a custom Gym Environment 
    Args:
    - sampleSize: int, number of residentials to be simulated, example - 1000
    - stepSize: int, step size of simulation, unit: minute, example - 15
    - simHorizon: tuple, (start_date, final_date), start and final date of simulation, 
        example - ('2020-07-01', '2020-08-01')
    - ambientWeather: pandas.dataframe, the time index needs to match with simulation horizon and step size, have two columns
        Temperature: ambient temperature, unit: degC
        CloudCoverage: cloud cover, optional
    - ttc: tuple, (ttc_mean, ttc_sigma), mean and standard deviation of thermal time constant, unit: h, example - (10.0, 5.0)
    - teq: tuple, (teq_mean, teq_sigma), mean and standard deviation of equivalent temperature increase due to other 
        heat gains (solar, occupant, equipment), unit: degC, example - (11.4, 5.7)
    - hvacMode: str, three options supported, 'heating only', 'cooling only', 'heating and cooling'
        # 'heating only': only the lower bound of comfort range (tsp-trange/2) is activated when calculating the comfort cost
        # 'cooling only': only the upper bound of comfort range (tsp+trange/2) is activated when calculating the comfort cost
        # 'heating and cooling': both the lower and upper bound of comfort range is activated
    - tsp: tuple, (tsp_mean,tsp_sigma), mean and standard deviation of initial temperature, unit: degC
    - trange: tuple, (trange_mean,trange_sigma), mean and standard deviation of acceptable temperature range, unit: degC
    - costWeight: tuple, (comfort_weight, energy_weight), weight to calculate the reward,
        comfort cost is defined as uncofmortable degree hours, unit K*H; 
        energy cost is defined as total HVAC energy, unit kWh
        more cost type and weight to be added
    - rcRatio: tuple, (r_c_ratio_mean, r_c_ratio_sigma), mean and standard deviation of R/C, unit: degC2/(kW*kWh)
        R and C are linearly correlate because they all corrected to the floor area 
    - copH: tuple, (COP_h_mean, COP_h_sigma), mean and standard deviation of heating COP
    - copC: tuple, (COP_c_mean, COP_c_sigma), mean and standard deviation of heating COP
    - teqHQ: tuple, (teq_q_mean, teq_q_sigma), mean and standard deviation of equivalent temperature of heating capacity, unit: degC
    - teqCQ: tuple, (teq_c_mean, teq_c_sigma), mean and standard deviation of equivalent temperature of cooling capacity, unit: degC
    - x0: tuple, (x0_mean, x0_sigma), mean and standard deviation of initial temperature, unit: degC
        if x0 is None, then the initial temperature is a random sampling from the uniform distribution between T_low and T_high
    - internalHGMethod: str, the method to calculate internal heat gains, 'DOE' and 'Ecobee' are supported
        'DOE': Use occupancy, lighting, and plug load schedule of DOE Reference Models, does not differentiate weekends and weekdays
        'Ecobee': Use the schedule inferred from Ecobee database, two clusters of occupancy are identified, different for weekends and weekdays 
    - internalHeatGainRatio: tuple, (internal_heat_gain_ratio_mean, internal_heat_gain_ratio_sigma), percentage of internal heat gain
        of total other heat gain (internal + solar)
    - noiseSigma: float, standard deviation of noise, model noise, could be result of unanticipated behaviors such as opening the windows, 
                   implication see paper https://doi.org/10.1016/j.enconman.2008.12.012
                   unit degC/s^0.5, default value - 0.02
    - measurementErrorSigma: float, standard deviation of measurement error, unit degC, default value - 0.5

    Action space:
    - Discrete action space: 0 for free floating, 1 for heating, 2 for cooling
    - The action space is not influenced by the hvac_mode
        In 'heating only' mode, action 2 could be accepted, but would not trigger cooling;
        In 'cooling only' mode, action 1 could be accepted, but would not trigger heating;
    Observation space:
    - Weather: The first two elements in obs are the ambient temperature, cloud coverage
    - Time: Hour of Day
    - Temperature: The remainings are temperature of each TCL
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, sampleSize, stepSize, simHorizon,
        ambientWeather, ttc, teq, 
        hvacMode, tsp, trange, costWeight=(10, 1), internalHGMethod='Ecobee',
        rcRatio=(0.7, 0.4),
        x0 = None,
        copH=(2.5,0.5), copC=(2.5,0.5), teqHQ=(50,10), teqCQ=(-50,10),
        internalHeatGainRatio=(0.3, 0.1), noiseSigma=0.03, measurementErrorSigma=0.5
        ):

        assert hvacMode in ['heating only', 'cooling only', 'heating and cooling'], \
            "HVAC Mode of {} is not supported. Please input 'heating only', 'cooling only' or 'heating and cooling'".format(hvacMode)
        assert internalHGMethod in ['DOE', 'Ecobee'], \
            "Internal heat gain method of {} is not supported. Please input 'DOE', or 'Ecobee'".format(internalHGMethod)

        self.sample_size = sampleSize
        self.step_size = stepSize
        self.sim_horizon = simHorizon
        self.ambient_weather = ambientWeather
        self.ttc = ttc                       # R*C
        self.r_c_ratio = rcRatio             # R/C
        self.hvac_mode = hvacMode
        self.cop_h = copH
        self.cop_c = copC
        self.teq_h_q = teqHQ
        self.teq_c_q = teqCQ
        self.x0 = x0
        self.tsp = tsp
        self.trange = trange
        self.teq = teq
        self.internalHG_method = internalHGMethod
        self.noise_sigma = noiseSigma
        self.measurement_error_sigma = measurementErrorSigma
        self.cost_weight = costWeight
        self.internal_heat_gain_ratio = internalHeatGainRatio    # Percentage of internal heat gain of teq

        self.episode_idx = 0

        # Check the weather file
        self.t_index = pd.date_range(start = self.sim_horizon[0], 
                                     end   = self.sim_horizon[1], 
                                     freq='{}T'.format(self.step_size))
        self.n_steps = len(self.t_index)

        assert self.n_steps+int(24*60/(self.step_size)) == len(self.ambient_weather), \
            'Number of entries for ambient temperature needs to equal to the number of timesteps plus one day'

        if 'Temperature' in self.ambient_weather.columns:
            self.x_amb = self.ambient_weather['Temperature']
            assert self.x_amb.isna().sum()==0, \
                'There are {} missing entries in the ambient temperature'.format(self.x_amb.isna().sum())
        else:
            raise KeyError('Temperature data is missing in the ambient weather file')

        if 'CloudCoverage' in self.ambient_weather.columns:
            self.cloud_amb = self.ambient_weather['CloudCoverage']
            assert self.cloud_amb.isna().sum()==0, \
                'There are {} missing entries in the ambient cloud coverage'.format(self.cloud_amb.isna().sum())
        else:
            ## TOCORRECT self.cloud_amb = 0
            print('Cloud coverage data is missing in the ambient weather file. We assume everyday is sunny')

        
        # Sampling
        self.Ttc = np.random.normal(*self.ttc, self.sample_size).clip(0.1,)
        self.Teq = np.random.normal(*self.teq, self.sample_size).clip(0.1, 18)
        self.R_c_ratio = np.random.normal(*self.r_c_ratio, self.sample_size).clip(0.1,)
        self.Cop_h = np.random.normal(*self.cop_h, self.sample_size).clip(0.1,)
        self.Cop_c = np.random.normal(*self.cop_c, self.sample_size).clip(0.1,)
        self.Teq_h_q = np.random.normal(*self.teq_h_q, self.sample_size).clip(min=0.1)  
        self.Teq_c_q = np.random.normal(*self.teq_c_q, self.sample_size).clip(max=-0.1) # negative value
        self.T_sp = np.random.normal(*self.tsp, self.sample_size)
        self.T_range = np.random.normal(*self.trange, self.sample_size).clip(0.1,)
        self.T_low = self.T_sp - self.T_range/2
        self.T_high = self.T_sp + self.T_range/2
        if self.x0:
            self.X_0 = np.random.normal(*self.x0, self.sample_size)
        else:
            self.X_0 = np.random.uniform(self.T_low, self.T_high)
        self.Internal_heat_gain_ratio = np.random.normal(*self.internal_heat_gain_ratio, self.sample_size).clip(0.1,)

        self.R = (self.Ttc*self.R_c_ratio)**0.5   # Unit: degC/kW
        self.C = self.Ttc/self.R                  # Unit: kWh/degC
        self.Q_h = self.Teq_h_q/self.R
        self.Q_c = self.Teq_c_q/self.R    # should be negative

        # Calculate heat dynamic coefficients
        self.A = np.exp(-(self.step_size/60)/self.Ttc)

        # Calculate the solar and internal heat gain schedule
        self.solHG_schd = get_solarHG_schd()

        # define the state and action space
        self.action_space = spaces.Discrete(3)  # 1 for heating, 2 for cooling, 0 for free floating
        self.obs_names = ['temp_amb', 'cloudCover_amb','hour'] + ['temp_{}'.format(n) for n in range(self.sample_size)]
        self.obs_low  = np.array([-30,  0,  0] + [0 for n in range(self.sample_size)])
        self.obs_high = np.array([50, 100, 23] + [35 for n in range(self.sample_size)])
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high, dtype=np.float16)

    def reset(self):
        self.episode_idx += 1
        self.time_step_idx = 0
        self.total_energy = 0  # Total energy
        self.total_UCDH = 0    # Total Uncomfortable Degree Hour
        self.obs = np.zeros(len(self.obs_names))
        self.action = np.zeros(len(self.obs_names)-1)
        self.obs[self.obs_names.index('temp_amb')] = self.x_amb[self.time_step_idx]
        self.obs[self.obs_names.index('cloudCover_amb')] = self.cloud_amb[self.time_step_idx]
        self.obs[self.obs_names.index('hour')] = self.t_index[self.time_step_idx].hour
        self.obs[-self.sample_size:] = self.X_0
        # Calculate other heat gains
        dayOfWeek = self.t_index[self.time_step_idx].dayofweek
        self.Teq_int = self._calc_internal_heat_gain(self.internalHG_method, dayOfWeek)
        self.Teq_sol_sunny = self._calc_solar_heat_gain()  # cloud coverage not considered

        return self.obs

    def step(self, action):
        # Convert the input action
        if self.hvac_mode == 'heating only':
            self.action = np.where(action==2, 0, action)    # turn off cooling
        elif self.hvac_mode == 'cooling only':
            self.action = np.where(action==1, 0, action)         # trun off heating

        # If heat gain method is 'Ecobee', resample to determine the cluster type.
        # Update internal heat gain at the begining of each day
        if self.internalHG_method == 'Ecobee':
           if self.time_step_idx%(24*60/self.step_size) == 0:
                dayOfWeek = self.t_index[self.time_step_idx].dayofweek
                self.Teq_int = self._calc_internal_heat_gain(self.internalHG_method, dayOfWeek)

        self.time_step_idx += 1
        # Update states: ambient and indoor temperatures
        self.obs[self.obs_names.index('temp_amb')] = self.x_amb[self.time_step_idx]
        self.obs[self.obs_names.index('cloudCover_amb')] = self.cloud_amb[self.time_step_idx]
        self.obs[self.obs_names.index('hour')] = self.t_index[self.time_step_idx].hour
        self._take_action()  
        reward = self._compute_reward()

        if self.time_step_idx < (self.n_steps - 1):
            done = False
        else:
            done = True
            self.render()
        
        return self.obs, reward, done, {'Energy': self.total_energy,
                                        'Comfort_UCDH': self.total_UCDH,
                                        'intHG': self.intHG, 'solHG': self.solHG, 
                                        'noise': self.model_noise, 'error': self.mea_error}
    
    def getParameters(self):
        '''
        Return a dataframe with parameter values of the environment.
        Each row is a sample of the households
        Each column is a parameter, columns include:
            R: thermal resistance
            C: thermal capacity
            P_h: heating nominal power, unit: kW electricity
            P_c: cooling nominal power, unit: kW electricity
            Q_h: heating capacity, unit kW heat
            Q_c: cooling capacity, unit kW heat
            COP_h: heating COP
            COP_c: cooling COP
            T_sp: temperature set point
            T_range: temperature acceptable range
        '''
        parameters = pd.DataFrame({
            'R': self.R,
            'C': self.C,
            'P_h': self.Q_h/self.Cop_h,
            'P_c': -self.Q_c/self.Cop_c,
            'Q_h': self.Q_h,
            'Q_c': self.Q_c,
            'COP_h': self.Cop_h,
            'COP_c': self.Cop_c,
            'T_sp': self.T_sp,
            'T_range': self.T_range,
        })
        return parameters

    def get_solarHG_schd(self):
        return self.solHG_schd

    def otherHGForecast(self):
        '''Hourly prediction of other heat gains (solar + internal) of the next 24 hours
        Assumption
            - internal heat gain of tomorrow is the same as today
            - cloud cover would always be 90%
        '''
        otherHG = self.Teq_int + self.Teq_sol_sunny*0.9
        otherHG = np.concatenate((otherHG, otherHG), axis=1) # 24 hours forecast extend to the next day
        hour_current = self.t_index[self.time_step_idx].hour
        otherHG_forecast = otherHG[:,hour_current:hour_current+24]
        return otherHG_forecast

    def weatherForecast(self):
        '''Hourly weather forecast of the next 24 hours
        '''
        weather_fore = self.ambient_weather[self.t_index[self.time_step_idx]:
                                            self.t_index[self.time_step_idx]+timedelta(days=1)]
        weather_fore_hourly = weather_fore.resample('H').mean()
        return weather_fore_hourly

    def _take_action(self):
        # Sample the noise
        self.model_noise = np.random.normal(0, self.noise_sigma*(self.step_size*60)**0.5, self.sample_size)
        temp_amb = self.obs[self.obs_names.index('temp_amb')]
        cloudCover_amb = self.cloud_amb[self.time_step_idx]
        hour_of_day = int(self.obs[self.obs_names.index('hour')])
        self.intHG = self.Teq_int[:,hour_of_day]
        self.solHG = self.Teq_sol_sunny[:,hour_of_day]*(1-cloudCover_amb/100)
        temp_zones = self.obs[-self.sample_size:]
        # Implement heating and cooling
        self.Q_input = np.where(self.action==1, self.Q_h,  0)
        self.Q_input = np.where(self.action==2, self.Q_c, self.Q_input)
        self.mea_error = np.random.normal(0, self.measurement_error_sigma, self.sample_size)

        self.obs[-self.sample_size:] = self.A*temp_zones + (1-self.A)*(temp_amb + self.intHG + self.solHG + self.model_noise + \
            self.R*self.Q_input) + self.mea_error

    def _calc_internal_heat_gain(self, method, day_of_week):
        '''Calculate the internal heat gain for the whole day
        '''
        Teq_intHG = self.Teq*self.Internal_heat_gain_ratio
        intHG_schd = get_intHG_schd(method, self.sample_size, day_of_week)
        Teq_int = Teq_intHG[:, np.newaxis]*intHG_schd
            # axis-0: households, axis-1: hour of day
        return Teq_int

    def _calc_solar_heat_gain(self):
        Teq_sol = self.Teq*(1-self.Internal_heat_gain_ratio)
        Teq_solar_sunny = Teq_sol[:, np.newaxis]*self.solHG_schd
            # axis-0: households, axis-1: hour of day
        return Teq_solar_sunny

    def _compute_reward(self):
        comfort_cost = np.clip(self.T_low-self.obs[-self.sample_size:], 0, None).sum()*(self.step_size/60) + \
            np.clip(self.obs[-self.sample_size:]-self.T_high, 0, None).sum()*(self.step_size/60)
        self.E_input = np.where(self.action==1, self.Q_h/self.Cop_h,  0)
        self.E_input = np.where(self.action==2, -self.Q_c/self.Cop_c,  self.E_input)   # self.Q_c is negative value
        energy_cost = self.E_input.sum()*(self.step_size/60)

        cost = comfort_cost*self.cost_weight[0] + energy_cost*self.cost_weight[1]
        self.total_energy += energy_cost
        self.total_UCDH += comfort_cost
        
        return cost*-1

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('Episode: {}'.format(self.episode_idx))
        print("Total Energy Consumption (kWh)")
        print(self.total_energy)
        print("Total Uncomfortable Degree Hours (K*h)")
        print(self.total_UCDH)