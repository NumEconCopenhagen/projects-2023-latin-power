import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import math

class HumanCapitalModelClass:

    def __init__(self):
        """ Setup model of private investment in education, with the aim to have more Human Capital.

        The model has two periods and a representative individual.
        
        The model depend on the following parameters:
        
        A: Initial level of ability (for instance IQ) that does not change over time
        E: Governamental expenditure in educatiion (for instance wage of teachers)
        H0: Initial level of human capital, which is homogeneous for all individuals
        B0: Return of education in terms of wages at time t
        B1: Return of education in terms of wages at time t+1
        rho: discount rate (for instance patience)
        gama: direct cost of education (for instance fee)
        delta: depreciatin of education (for instance losing capacities or obsolences)
        alpha: productivity of education accumulation
        """

        # a. create namespaces
        par = self.par = SimpleNamespace() #group of variables that now is empty
        sol = self.sol = SimpleNamespace()

        # b. parameters influencing decision making
        par.A = 1.0
        par.E = 1.0
        par.H0 = 1.0
        par.B0 = 0.5 #Present and future wages are equal
        par.B1 = 0.5
        par.rho = 0.0025 #Low patience
        par.gama= 1.0
        par.delta=0.0 #Human Capital does not change over time
        par.alpha=0.5

        # c. household production
        #par.alpha = 0.5
        #par.sigma = 1.0

        # d. wages
        #par.wM = 1.0
        #par.wF = 1.0
        #par.wF_vec = np.linspace(0.8, 1.2, num=5, endpoint=True)

        # e. targets
        #par.beta0_target = 0.4
        #par.beta1_target = -0.1

        # f. solution
        #sol.LM_vec = np.zeros(par.wF_vec.size)
        #sol.HM_vec = np.zeros(par.wF_vec.size)
        #sol.LF_vec = np.zeros(par.wF_vec.size)
        #sol.HF_vec = np.zeros(par.wF_vec.size)

        #sol.beta0 = np.nan
        #sol.beta1 = np.nan
        
        #g. time
        #par.nm = 100 # continously

    def calc_utility(self,S0,S1):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. human capital production function
        H1 = par.H0*(1-par.delta)+(par.A*par.H0*par.E*S0)**par.alpha

        # b. benefit of schooling in time t
        u0 = par.B0*par.H0*(1-S0) 

        # c. cost of schooling in time t
        c0=par.gama*S0

        # d. benefit of schooling in time t+1
        u1 = par.B1*H1*(1-S1)

        # e. cost of schooling in time t+1
        c1=par.gama*S1

        # c. total consumption utility
        #Q = C**par.omega*H**(1-par.omega)
        #utility = (np.fmax(Q,1e-8)**(1-par.rho))/(1-par.rho)

        # d. disutlity of work
        #epsilon_ = 1+1/par.epsilon
        #TM = LM+HM
        #TF = LF+HF
        #disutility = par.nu*((TM**epsilon_)/epsilon_+(TF**epsilon_)/epsilon_)
        
        return u0 - c0 + (1/(1+par.rho))*(u1 - c1)  ##the utility function

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,1,49)
        S0,S1 = np.meshgrid(x,x) # all combinations
    
        S0 = S0.ravel() # vector
        S1 = S1.ravel()

        # b. calculate utility
        v = self.calc_utility(S0,S1)
    
        # c. set to minus infinity if constraint is broken
        I = (S0 > 1) | (S1 > 1) # | is "or"
        v[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(v)
        
        opt.S0 = S0[j]
        opt.S1 = S1[j]
        opt.v = v[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve_continously(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. calculate utility with negative since we will use minimize()
        def u(x):
             return -self.calc_utility(x[0],x[1])

        # b. constraints and bounds
        bounds = optimize.Bounds([0, 0],[1, 1])
        #linear_constraint = optimize.LinearConstraint([[ 0, 0], [0, 0, 1, 1]], [0, 0], [25, 25])

        # c. initial guess
        x_guess = np.array([0.5,0.5])

        # d. find maximization
        ans = optimize.minimize(u, x_guess, method="SLSQP", bounds=bounds)

        opt.S0 = ans.x[0]
        opt.S1 = ans.x[1]
        opt.v = -ans.fun

        # e. print answer
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        # print(ans.message)
        # print(f'LM = {ans.x[0]:.0f}, HM = {ans.x[1]:.0f}, LF = {ans.x[2]:.0f}, HF = {ans.x[3]:.0f}, Utility = {ans.fun:.4f}')
        return opt
    