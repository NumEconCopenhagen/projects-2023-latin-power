import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy import optimize
import math

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace() #group of variables that now is empty
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8, 1.2, num=5, endpoint=True)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan
        
        #g. time
        par.nm = 100 # continously

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production  #UPDATED ALE
        if par.sigma == 0.0 :
            H=min(HM,HF)
        elif par.sigma == 1.0 :
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = (1-par.alpha)*HM**(1-1/par.sigma) + (par.alpha)*HF**(1-1/par.sigma)


        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility  ##the utility function

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    
    def solve_continously(self,do_print=True):
        """ solve model continously """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
         # a. calculate utility with negative since we will use minimize()
        def u(x):
            return -self.calc_utility(x[0],x[1],x[2],x[3])
        
        # b. constraints and bounds
        bounds = optimize.Bounds([0, 0, 0, 0],[25, 25, 25, 25])
        linear_constraint = optimize.LinearConstraint([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 0], [25, 25])
        
        # c. initial guess
        x_guess = np.array([0,0,0,0])
        
        # d. find maximization
        ans = optimize.minimize(u, x_guess,method='trust-constr', bounds=bounds, constraints=linear_constraint)
        
        opt.LM = ans.x[0]
        opt.HM = ans.x[1]
        opt.LF = ans.x[2]
        opt.HF = ans.x[3]
        opt.u = ans.fun
        
        # e. print answer
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        # print(ans.message)
        # print(f'LM = {ans.x[0]:.0f}, HM = {ans.x[1]:.0f}, LF = {ans.x[2]:.0f}, HF = {ans.x[3]:.0f}, Utility = {ans.fun:.4f}')

        return opt

    def solve_wF_vec(self,discrete=False,do_print=False):
        """ solve model for vector of female wages """
        
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        dic_sol_q4 = {}
        
        for iterator in range(0, self.par.wF_vec.size, 1):  ## solving the model for each value of wage
            self.par.wF=self.par.wF_vec[iterator]
            opt = self.solve_continously()
            # print("iteration =", iterator, "wage of woman", self.par.wF, "sigma = ", self.par.sigma, "alpha = ", self.par.alpha)
  
            sol.LM_vec[iterator]=(opt.LM)
            sol.HM_vec[iterator]=(opt.HM)
            sol.LF_vec[iterator]=(opt.LF)
            sol.HF_vec[iterator]=(opt.HF)
            
            dic_sol_q4[iterator] ={'wF': self.par.wF, 'wM': self.par.wM, 'LM': opt.LM, 'HM': opt.HM, 'LF': opt.LF, 'HF': opt.HF, 'logr_HF_HM': math.log(opt.HF/opt.HM), 'logr_wF_wM': math.log(self.par.wF/self.par.wM)}
            
            if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
                
        return dic_sol_q4
    

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass