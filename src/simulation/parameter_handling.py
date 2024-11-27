import copy
import numpy as np
import pandas as pd
import warnings


### CODE TO HANDLE INPUT PARAMETERS ###

# to contain the input parameters for the simulation
class InputParams:
    def __init__(self, seq_data, env_data, dsb_data):
        # unpack parameters
        (self.seq_name, self.sequence, self.seq_list, self.types) = seq_data
        (self.T, self.I, self.Hc, self.Nc, self.Nt, self.Cc, self.Ct) = env_data
        (self.dsb_indices) = dsb_data
        # store original values for when sequence is later modified
        self.set_original(copy.deepcopy((self.sequence, self.seq_list)))
    @classmethod
    def read_files(cls, seq_file, env_file, dsb_file):
        # create class instance to return
        new = cls()
        # read FASTA file
        f = open(seq_file, 'r').readlines()
        seq_name = f[0][1:].strip()
        sequence = f[1].strip()
        seq_list = list(new.sequence)
        types = list(np.unique(seq_list)) # get unique residue types
        f.close()
        # read physicochemical parameters
        T, I, Hc, Nc, Nt, Cc, Ct = np.loadtxt(env_file, unpack=True)
        # read disulfide bond parameters
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dsb_indices = np.loadtxt(dsb_file, ndmin=2)
        # package variables
        seq_data = (seq_name, sequence, seq_list, types)
        env_data = (T, I, Hc, Nc, Nt, Cc, Ct)
        dsb_data = (dsb_indices)
        return cls(seq_data, env_data, dsb_data)
    def copy(self, deep=True):
        if deep:
            seq_data_copy = copy.deepcopy((self.seq_name, self.sequence,
                                           self.seq_list, self.types))
            env_data_copy = copy.deepcopy((self.T, self.I, self.Hc,
                                           self.Nc, self.Nt,
                                           self.Cc, self.Ct))
            dsb_data_copy = copy.deepcopy((self.dsb_indices))
            original_copy = copy.deepcopy((self.sequence_original,
                                           self.seq_list_original))
        else:
            seq_data_copy = (self.seq_name, self.sequence,
                             self.seq_list, self.types)
            env_data_copy = (self.T, self.I, self.Hc,
                             self.Nc, self.Nt,
                             self.Cc, self.Ct)
            dsb_data_copy = (self.dsb_indices)
            original_copy = (self.sequence_original,
                             self.seq_list_original)
        new = InputParams(seq_data_copy, env_data_copy, dsb_data_copy)
        new.set_original(original_copy)
        return new
    def set_original(self, seq_data_original):
        self.sequence_original = seq_data_original[0]
        self.seq_list_original = seq_data_original[1]
    def save_text(self, seq_file, env_file, dsb_file):
        f = open(seq_file, 'w')
        f.write('>{:s}\n{:s}'.format(self.seq_name, self.sequence))
        f.close()
        to_save = np.array([self.T, self.I, self.Hc,
                            self.Nc, self.Nt,
                            self.Cc, self.Ct])
        header = ('temperature, ionic_strength, His_charge_proportion, ' +
                  'N_term_charged, N_term_truncated, ' +
                  'C_term_charged, C_term_truncated')
        np.savetxt(env_file, to_save.T, header=header)
        np.savetxt(dsb_file, np.array(dsb_indices))

# update sequence and residue parameters based on specified charge information
# returns an updated InputParams object and DataFrame of residue parameters
# the new InputParams object will still also the original seq info (_original)
def update_params(input_params, res_params):
        # make local copies of residue parameters and sequence
        ip_new = input_params.copy(deep=True)
        rp_new = res_params.copy(deep=True)
        # create parameters for charged or modified termini
        if input_params.Nc == 1:
            rp_new.loc['X'] = rp_new.loc[input_params.sequence[0]]
            rp_new.loc['X', 'MW'] += 2
            rp_new.loc['X', 'q'] += 1
            ip_new.seq_list[0] = 'X'
        elif input_params.Nt != 1:
            rp_new.loc['X'] = rp_new.loc[input_params.sequence[0]]
            rp_new.loc['X', 'MW'] += 43
            ip_new.seq_list[0] = 'X'
        if input_params.Cc == 1:
            rp_new.loc['Z'] = rp_new.loc[input_params.sequence[-1]]
            rp_new.loc['Z', 'MW'] += 16
            rp_new.loc['Z', 'q'] -= 1
            ip_new.seq_list[-1] = 'Z'
        elif input_params.Ct != 1:
            rp_new.loc['Z'] = rp_new.loc[input_params.sequence[-1]]
            rp_new.loc['Z', 'MW'] += 16
            ip_new.seq_list[-1] = 'Z'
        # set the histidine charge
        rp_new.loc['H', 'q'] = input_params.Hc
        # update sequence and types
        ip_new.sequence = ''.join(ip_new.seq_list)
        ip_new.types = list(np.unique(ip_new.seq_list))
        return ip_new, rp_new


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
