import numpy as np
import itertools
import pandas as pd

### CODE TO CALCULATE SEQUENCE PROPERTIES ###

def calc_seq_prop(input_params, res_params):

    ### LOCAL COPIES AND SEQUENCE LENGTH ###
    ip = input_params.copy(deep=True)
    rp = res_params.copy(deep=True)
    N = len(ip.sequence)

    ### GET PAIRS AND SEPARATIONS ###
    # get residue pairs
    pairs = np.array(list(itertools.combinations(ip.seq_list_original, 2)))
    pairs_indices = np.array(list(itertools.combinations(range(N), 2)))
    # calculate sequence separations
    ij_dist = np.diff(pairs_indices, axis=1).flatten().astype(float)

    ### CALCULATE CHARGE-INDEPENDENT PROPERTIES ###
    # calculate residue fractions
    fK = sum([ip.seq_list_original.count(a) for a in ['K']])/N
    fR = sum([ip.seq_list_original.count(a) for a in ['R']])/N
    fE = sum([ip.seq_list_original.count(a) for a in ['E']])/N
    fD = sum([ip.seq_list_original.count(a) for a in ['D']])/N
    fARO = sum([ip.seq_list_original.count(a) for a in ['F', 'W', 'Y']])/N
    # calculate mean lambda
    mean_lambda = np.mean(res_params.loc[ip.seq_list_original].lambdas)
    # calculate lambda sums, needed for SHD
    ll = rp.lambdas.loc[pairs[:,0]].values + rp.lambdas.loc[pairs[:,1]].values
    # calculate SHD
    beta = -1
    shd = np.sum(ll*np.power(np.abs(ij_dist), beta))/N
    # calculate Omega_ARO
    seq = SequenceParameters(ip.sequence_original)
    omega = seq.get_kappa_X(grp1=['F', 'Y', 'W'])

    ### CALCULATE CHARGE-DEPENDENT PROPERTIES ###
    # create a modified sequence for kappa calculation
    seq_kappa = np.array(ip.seq_list_original)
    # replace His and termini with 'A', 'K', or 'D' depending on charge
    H_replace = 'K' if Hc >= 0.5 else 'A'
    seq_kappa[np.where(np.array(ip.seq_list_original) == 'H')[0]] = H_replace
    if Nc == 1: seq_kappa[0] = 'K' if rp.loc['X', 'q'] > 0 else 'A'
    if Cc == 1: seq_kappa[-1] = 'D' if rp.loc['Z', 'q'] < 0 else 'A'

    # calculate properties that depend on charges
    pairs = np.array(list(itertools.combinations(ip.seq_list, 2)))
    # calculate charge products
    qq = rp.q.loc[pairs[:,0]].values*rp.q.loc[pairs[:,1]].values
    # calculate SCD
    scd = np.sum(qq*np.sqrt(ij_dist))/N
    kappa = SequenceParameters(''.join(seq_kappa)).get_kappa()
    fcr = rp.q.loc[ip.seq_list].abs().mean()
    ncpr = rp.q.loc[ip.seq_list].mean()

    return np.around([fK, fR, fE, fD, fARO, mean_lambda,
                      shd, omega, scd, kappa, fcr, ncpr],3)
