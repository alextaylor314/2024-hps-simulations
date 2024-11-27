import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import os
import pandas as pd
from scipy import linalg
from scipy.optimize import curve_fit

### CODE FOR GLOBAL CONFORMATIONAL ANALYSIS ###

# calculate gyration tensor to determine Rg, asphericity, oblateness
def gyr_tensor(traj, input_params, res_params):
    masses = res_params.loc[input_params.seq_list, 'MW'].values
    # calculate the centre of mass
    cm = np.sum(traj.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate distances from the centre of mass
    si = traj.xyz - cm[:,np.newaxis,:]
    q = np.einsum('jim,jin->jmn', si*masses.reshape(1,-1,1), si)/masses.sum()
    trace_q = np.trace(q,axis1=1,axis2=2)
    # calculate Rg
    Rg_array = ArrayAverage(np.sqrt(trace_q), use_blocking=True)
    # calculate traceless matrix
    mean_trace = trace_q/3
    q_hat = q - mean_trace.reshape(-1,1,1)*np.identity(3).reshape(-1,3,3)
    # calculate asphericity
    D_array = ArrayAverage(3/2*np.trace(q_hat**2,axis1=1,axis2=2)/(trace_q**2),
                           use_blocking=True)
    # calculate oblateness
    S_array = ArrayAverage(27*linalg.det(q_hat)/(trace_q**3), use_blocking=True)
    return Rg_array, D_array, S_array

# calculate inter-residue distances and fit the Flory exponent
def calc_Rij(traj):
    # get residue pairs and distances
    n_res = traj.n_atoms
    pairs = traj.top.select_pairs('all','all')
    d = md.compute_distances(traj, pairs)
    # calculate average distances for different separations
    ij = np.arange(2,n_res,1)
    diff = [x[1]-x[0] for x in pairs]
    dij = np.empty(0)
    for i in ij:
        dij = np.append(dij, np.sqrt((d[:,diff==i]**2).mean()))
    # fit to power law to obtain Flory exponent
    f = lambda x, R0, v : R0*np.power(x,v)
    popt, pcov = curve_fit(f, ij[ij>5], dij[ij>5], p0=[.4,.5])
    R0 = FittedQuantity(popt[0], pcov[0,0]**0.5)
    nu = FittedQuantity(popt[1], pcov[1,1]**0.5)
    Flory_data = FloryData(ij, dij, R0, nu)
    return Flory_data

# estimate Rh using the Kirkwood-Riseman equation
def estim_Rh(traj):
    pairs = traj.top.select_pairs('all','all')
    d = md.compute_distances(traj, pairs)
    diff = [x[1]-x[0] for x in pairs]
    mean_recip = [np.mean(i) for i in np.reciprocal(d[:,diff!=0])]
    harmonic = np.reciprocal(mean_recip)
    Rh_KR = ArrayAverage(harmonic, use_blocking=True)
    return Rh_KR

# calculate end-to-end distances for each frame
def calc_Ree(traj):
    pairs = np.array([[ 0,  len(list(traj.top.atoms))-1]])
    d = md.compute_distances(traj, pairs)
    Ree = ArrayAverage(d[...,0], use_blocking=True)
    return Ree

# run the global conformational analysis
def global_conformation(input_params, res_params, proj_dir,
                        plot_width, plot_height, save=False, show=False):

    ### LOAD TRAJECTORY AND CALCULATE PROPERTIES ###
    traj = md.load_dcd(proj_dir + '/SIMULATION/traj.dcd',
                       top = proj_dir + '/SIMULATION/top.pdb')
    Rg, D, S = gyr_tensor(traj, input_params, res_params)
    Flory_data = calc_Rij(traj) # Flory scaling exponent fit
    Rh = estim_Rh(traj) # estimate Rh using Kirkwood-Riseman equation
    Ree = calc_Ree(traj) # end-to-end distance

    ### CREATE PLOTS ###
    mpl.rcParams.update({'font.size': 7})
    fig, axs = plt.subplots(2, 3, figsize=(plot_width, plot_height),
                            facecolor='w', dpi=300, layout='constrained')
    axs = axs.flatten()
    # will loop through these variables
    objects = [Rg, Ree, Rh, D, S, Flory_data]
    x_names = [r'$R_g$ / nm', r'$R_{ee}$ / nm', r'$R_{h,KR}$ / nm',
               r'$\Delta$', r'$S$', r'$|i-j|$']
    y_names = [r'$p(R_g)$', r'$p(R_{ee})$', r'$p(R_{h,KR})$',
               r'$p(\Delta$)', r'$p(S)$',
               r'$\sqrt{\langle R_{ij}^2 \rangle}$  /  nm']
    # iterate through variables simultaneously
    to_iterate = zip(objects, x_names, y_names)
    for i, (obj, x_name, y_name) in enumerate(to_iterate):
        obj.create_plot(axs[i], x_name, y_name)

    ### CREATE DATAFRAMES ###
    # basic summary stats
    stats = np.c_[Rg.get_str_array(), Ree.get_str_array(), Rh.get_str_array(),
                  D.get_str_array(), S.get_str_array(),
                  Flory_data.nu.get_str_array()]
    stats_cols = ['Rg / nm', 'Ree / nm', 'Rh_KR / nm', 'Delta', 'S', 'nu']
    stats_rows = ['value', 'error', 'sigma']
    stats_df = pd.DataFrame(data=stats, columns=stats_cols, index=stats_rows)
    # time series of summary stats that can be calculated for single frames
    time_series = np.c_[Rg.array, Ree.array, Rh.array, D.array, S.array]
    time_series_cols = ['Rg / nm', 'Ree / nm', 'Rh_KR / nm', 'Delta', 'S']
    time_series_df = pd.DataFrame(data = time_series,
                                  columns = time_series_cols)
    # scaling profile used to calculate the Flory exponent
    dij_fit = Flory_data.R0.value*np.power(Flory_data.ij, Flory_data.nu.value)
    scaling_profile = np.c_[Flory_data.ij, Flory_data.dij, dij_fit]
    scaling_profile_cols = ['ij', 'Rij / nm', 'Rij_fit / nm']
    scaling_profile_df = pd.DataFrame(data = scaling_profile,
                                      columns = scaling_profile_cols)

    ### SAVE DATA IF REQUESTED ###
    if save:
        dir = proj_dir + '/global_conformation'
        try:
            os.mkdir(dir)
        except:
            pass
        plt.savefig(dir + '/global_conformation.pdf', dpi=300, facecolor='w',
                    edgecolor='w', orientation='portrait', bbox_inches='tight')
        stats_df.to_csv(dir + '/global_conformation_stats.csv')
        time_series_df.to_csv(dir + '/global_conformation_time_series.csv')
        scaling_profile_df.to_csv(dir + '/scaling_profile.csv')
        print('Saved conformational data.')

    ### SHOW PLOTS IF REQUESTED ###
    if show:
        plt.show()
        display(stats_df)
    else:
        plt.close()
