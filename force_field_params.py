import numpy as np
import pandas as pd

### CODE TO GENERATE FORCE FIELD PARAMETERS ###

# to generate and contain the Debye-Huckel parameters
class DH_Params:
    def __init__(self, input_params, res_params):
        fasta = input_params.seq_list
        # thermal energy scale
        kT = 8.3145*input_params.T*1e-3
        # calculate prefactor for the Yukawa potential
        f = lambda T: 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
        epsw = f(input_params.T)
        lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
        self.epsilon = [res_params.loc[a].q*np.sqrt(lB*kT) for a in fasta]
        # calculate inverse of the Debye length
        self.kappa = np.sqrt(8*np.pi*lB*input_params.I*6.022/10)

# to generate and contain the Lennard-Jones parameters
class LJ_Params:
    def __init__(self, input_params, res_params):
        # standard value of epsilon for the force field
        self.epsilon = 0.8368
        # calculate average sigma values for pairs of interacting residues
        sigma_val = res_params.sigmas.values
        sigma_ind = res_params.sigmas.index
        self.sigmas = pd.DataFrame((sigma_val + sigma_val.reshape(-1,1))/2,
                                index=sigma_ind, columns=sigma_ind)
        # calculate average lambda values for pairs of interacting residues
        lambda_val = res_params.lambdas.values
        lambda_ind = res_params.lambdas.index
        self.lambdas = pd.DataFrame((lambda_val + lambda_val.reshape(-1,1))/2,
                                index=lambda_ind, columns=lambda_ind)
