import numpy as np
import cvxpy as cp


class Controller():
    def __init__(self, T, dt, **kwargs):
        '''
        dt: planning timestep
        T: planning horizon
        **kwargs: Model Parameters
        '''
        self.T = T
        self.dt = dt
        self.err_count = 0
        
        self.R = kwargs["R"]
        self.C = kwargs["C"]
        self.Pm = kwargs["Pm"]
        self.eta = kwargs["eta"]
        self.T_sp = kwargs["theta"]
        self.Delta = kwargs["Delta"]
        self.sign = kwargs["sign"] #(+) for heating and (-) for cooling
            
        # Variable
        self.u = cp.Variable(T)
        
        # Save u_i-u_bar from previous time step
        self.u_diff = cp.Parameter(T)  # load rather than 0/1
        self.v_bar = cp.Parameter(T)
        self.w_bar = cp.Parameter(T)
        self.objective = cp.sum_squares(self.u-self.u_diff-self.v_bar+self.w_bar)

        ## Info needed for constraints
        self.X0 = cp.Parameter(T)
        self.D = cp.Parameter(T)
            
        # Set default value for constraints
        self.u_lower = cp.Parameter(T)
        self.u_lower.value = np.tile(0, T)
        self.u_upper = cp.Parameter(T)
        self.u_upper.value = np.tile(self.Pm, T)
        self.x_lower = cp.Parameter(T)
        self.x_lower.value = np.tile(self.T_sp-self.Delta, T)
        self.x_upper = cp.Parameter(T)
        self.x_upper.value = np.tile(self.T_sp+self.Delta, T)
        
        # Set default value for exogenous variables
        self.a = np.exp(-self.dt/(self.R*self.C))
        self.b = self.eta * self.R
        
        A = np.eye(T)
        for i in range(1,T):
            A[i, i-1] = self.a
        A_inv = np.linalg.inv(A)
        B = -np.eye(T)*self.b*(1-self.a)
            
        # Constraints
        self.constraints = [-self.u <= -self.u_lower,
                             self.u <= self.u_upper]
        # X_T = A_inv@(X0*a + B@self.u + D*(1-a))
        self.constraints += [-A_inv@(self.X0*self.a + B@self.u + self.D*(1-self.a)) <= -self.x_lower,
                              A_inv@(self.X0*self.a + B@self.u + self.D*(1-self.a)) <= self.x_upper]

        self.Problem = cp.Problem(cp.Minimize(self.objective),
                                  self.constraints)

    def u_update(self, v_bar, w_bar):
        self.v_bar.value = v_bar
        self.w_bar.value = w_bar
        try:
            self.Problem.solve()
        except:
            print("Solver failed")
            self.u.value = None
            
        ## Check solution valid
        if self.u.value is not None:
            return self.u.value, self.Problem.status
        else:
            u  = (self.X0.value[0]-self.T_sp)/self.Delta
            self.err_count += 1
            return np.ones(self.T)*np.clip(u, 0, 1)*self.Pm, self.Problem.status


    def updateState(self, x_init, u_lower = None, u_upper = None,
                    x_lower = None, x_upper = None,
                    t_amb = None): #
        '''Update initial state (temperature), 
        ambient temperature 
        
        '''
        # Update the initial temperature
        X_init = np.zeros(self.T)
        X_init[0] = x_init
        self.X0.value = X_init

        ## Update the exog Variables
        if t_amb is not None:
            assert len(t_amb) == self.T
            self.D.value = t_amb

        # Update constraints if necessary
        if u_lower is not None:
            if isinstance(u_lower, int) | isinstance(u_lower, float):
                self.u_lower.value = np.tile(u_lower, self.T)
            else:
                assert len(u_lower) == self.T
                self.u_lower.value = u_lower
        if u_upper is not None:
            if isinstance(u_upper, int) | isinstance(u_upper, float):
                self.u_upper.value = np.tile(u_upper, self.T)
            else:
                assert len(u_upper) == self.T
                self.u_upper.value = u_upper
        if x_lower is not None:
            assert len(x_lower) == self.T
            self.x_lower.value = x_lower
        if x_upper is not None:
            assert len(x_upper) == self.T
            self.x_upper.value = x_upper
            self.T_sp = (x_upper[0]+x_lower[0])/2
            self.Delta = (x_upper[0]-x_lower[0])/2
            

class ControllerGroup():
    def __init__(self, T, dt, parameters):
        '''
        parameters: dataframe, parameters for all TCLs, 
            row for each TCL, 
            column for each parameter, R, C, Pm, eta, theta, Delta, sign 
        '''
        self.parameters = parameters
        self.n_agent = len(self.parameters)
        self.T = T
        self.controller_list = []
        for idx in range(self.n_agent):
            self.controller_list.append(Controller(T = T, dt = dt, **self.parameters.loc[idx].to_dict()))
    
    def updateState(self, x_list, u_list = None, d_list = None, x_lower_list = None, x_upper_list = None):
        
        if u_list is not None:
            # convert from 0/1 to power
            assert u_list.shape == (self.n_agent, self.T)
            p_list = u_list*self.parameters['Pm'].values[:,np.newaxis]
            u_bar = np.mean(p_list, axis = 0)        
        
        for idx, controller in enumerate(self.controller_list):
            '''
            u_list: on/off
            '''
            controller.updateState(x_list[idx], 
                                   t_amb = d_list[idx] if d_list is not None else None, 
                                   x_lower = x_lower_list[idx] if x_lower_list is not None else None, 
                                   x_upper = x_upper_list[idx] if x_upper_list is not None else None)
            
            ## Initialize the controller with action from prev timestep
            if u_list is not None:
                controller.u_diff.value = p_list[idx] - u_bar
            else:
                controller.u_diff.value = np.zeros(self.T)
                
    def u_update(self, v_bar, w_bar):
        u_list = []
        #print("v_bar", v_bar)
        #print("w_bar", w_bar.shape)
        for idx, controller in enumerate(self.controller_list):
            #print(idx)
            u_i, status = controller.u_update(v_bar, w_bar)
            if status in ["infeasible", "unbounded"]:
                print(idx, status)
            u_list.append(u_i)
        
        u_bar = np.mean(u_list, axis = 0)
        for idx, controller in enumerate(self.controller_list):
            controller.u_diff.value = u_list[idx] - u_bar
        return u_bar, np.array(u_list)
        
class Aggregator():
    def __init__(self, T, g_func, param_shapes, n_agent, rho):
        self.T = T
        self.N = n_agent
        self.rho = rho
        self.v_bar = cp.Variable(self.T)
        
        self.u_bar = cp.Parameter(self.T)
        self.w_bar = cp.Parameter(self.T)
        
        self.param_list = []
        for item in param_shapes:
            self.param_list.append(cp.Parameter(item))
        #print(self.param_list)
        
        # Objective:
        cost = g_func(self.N*self.v_bar, self.param_list)
        
        cost += self.N*self.rho/2*cp.sum_squares(self.u_bar-self.v_bar+self.w_bar)
        self.Problem = cp.Problem(cp.Minimize(cost))

    def setValues(self, value_list):
        assert len(self.param_list) == len(value_list)
        for idx, item in enumerate(self.param_list):
            item.value = value_list[idx]

    def v_update(self, u_bar, w_bar):
        self.u_bar.value = u_bar
        self.w_bar.value = w_bar
        self.Problem.solve()
        return self.v_bar.value
   

def g_tracking(v_agg, variable_list):
    return cp.sum_squares(v_agg - variable_list[0])
   
def ADMM(controllers, aggregator, target_list, eps_primal, eps_dual, v_init = None, max_iter = 100):
    N = controllers.n_agent
    T = controllers.T
    rho = aggregator.rho

    if v_init is not None:
        v_bar = v_init
    else:
        v_bar = np.zeros(T)
    w_bar = np.zeros(T)

    u_bar_prev = np.zeros(T)
    u_prev = np.zeros((N, T))
    v_bar_prev = v_bar

    converged = False
    n_iter = 0

    aggregator.setValues(target_list)

    while (not converged) & (n_iter<100):
        # u: N x T
        u_bar, u = controllers.u_update(v_bar, w_bar)
        v_bar = aggregator.v_update(u_bar, w_bar)
        w_bar += u_bar - v_bar
        
        # For checking convergence
        r = u_bar - v_bar
        s = rho*(u_bar.reshape(1, -1)-u_bar_prev.reshape(1, -1)
                 -(u-u_prev)-(v_bar.reshape(1, -1)-v_bar_prev.reshape(1, -1)))
        u_bar_prev = u_bar
        u_prev = u
        v_bar_prev = v_bar
        
        n_iter+=1
        if (N*np.linalg.norm(r, 2)<eps_primal) & (sum(np.linalg.norm(s, 2, axis = 0))<eps_dual):
            converged = True
    return u, n_iter
