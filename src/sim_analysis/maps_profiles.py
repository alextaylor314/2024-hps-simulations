import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import os
import pandas as pd

### CODE FOR GENERATING INTER-RESIDUE MAPS AND PROFILES ###

'''
Creates lists to handle padding and masking.
Padding:  Spaces will be added in the energy maps and profiles, eg. to account
          for missing regions.
Masking:  Interactions involving a given region will be masked when calculating
          energy profiles.
Format:   res(start-end) for unmasked residues
          mask(start-end) for masked residues
          pad(length) for padding
          Build sequence regions in order, separated with commas.
          eg. res(1-40), pad(30), res(41-50), mask(51-60), res.(61-80)
Returns:  map_structure is a list giving the order in which residues are to be
          displayed, and profile_mask is an array of 1 or 0 used to decide which
          interactions are included in summations for the profiles
'''
def get_pad_mask(user_input):
    spaces_removed = ''.join(user_input.split())
    commands_list = spaces_removed.split(',')
    map_structure = []
    profile_mask = []
    for command in commands_list:
        try:
            command_type = command.split('(')[0]
            command_args = command.split('(')[1].replace(')', '')
            if '-' in command_args:
                arg_list = [int(i) for i in command_args.split('-')]
            else:
                arg_list = [int(command_args)]
            if command_type == 'res':
                for index in range(arg_list[0]-1, arg_list[1]):
                    map_structure.append(index)
                    profile_mask.append(1)
            elif command_type == 'mask':
                for index in range(arg_list[0]-1, arg_list[1]):
                    map_structure.append(index)
                    profile_mask.append(0)
            elif command_type == 'pad':
                for index in range(arg_list[0]):
                    map_structure.append(np.nan)
                    profile_mask.append(0)
        except:
            error_string = 'Incorrect mask/pad format: {}'.format(command_type)
            raise ValueError(error_string)
    return map_structure, profile_mask

def analyse_contacts(traj, input_params, res_params, frame_list=None,
                     ij_threshold=2, d_threshold=0.8, pad_mask=None):
    # get residue pairs
    n_res = traj.n_atoms
    pairs = traj.top.select_pairs('all','all')
    # exclude bonded pairs
    mask = np.abs(pairs[:,0]-pairs[:,1]) >= int(ij_threshold)
    for dsb_pair in input_params.dsb_indices:
        for n,p in enumerate(pairs):
            if p[0] == dsb_pair[0] and p[1] == dsb_pair[1]:
                mask[n] = False
                break
    pairs = pairs[mask]
    # get distances of residue pairs
    d = md.compute_distances(traj, pairs)
    if frame_list is not None:
        d = d[frame_list]
    # get contact frequencies
    contact_frames = np.where(d <= float(d_threshold), 1.0, 0.0)
    contact_freq = np.nansum(contact_frames, axis=0)/contact_frames.shape[0]
    # get lists to account for padding and masking
    if pad_mask is None:
        map_structure = range(n_res)
        profile_mask = [1]*n_res
    else:
        map_structure, profile_mask = get_pad_mask(pad_mask)
    map_dim = len(map_structure)
    map_positions = []
    for i in range(n_res):
        to_add = [j for j, element in enumerate(map_structure) if element == i]
        map_positions.append(to_add)
    # create contact map
    contact_map = pd.DataFrame(index=range(map_dim), columns=range(map_dim),
                               dtype=float)
    for k,(i,j) in enumerate(pairs):
        for i_pos in map_positions[i]:
            for j_pos in map_positions[j]:
                contact_map.loc[i_pos,j_pos] = contact_freq[k]
                contact_map.loc[j_pos,i_pos] = contact_freq[k]
    # create contact profiles
    mask_list = [i for i, keep in enumerate(profile_mask) if not keep]
    contact_map_masked = contact_map.copy(deep=True)
    contact_map_masked.loc[mask_list,:] = np.nan
    contact_map_masked.loc[:,mask_list] = np.nan
    contact_profile_array = np.array(contact_map_masked.sum(axis=1))
    contact_profile_array[mask_list] = np.nan
    contact_profile = ArrayAverage(contact_profile_array)
    # package and return
    return MapProfile(contact_map, contact_profile)

# generate energy maps and profiles: Ashbaugh-Hatch, Debye-Huckel, and combined
def analyse_energies(traj, input_params, res_params, frame_list=None,
                     pad_mask=None):
    # get residue pairs
    n_res = traj.n_atoms
    pairs = traj.top.select_pairs('all','all')
    # exclude bonded pairs
    mask = np.abs(pairs[:,0]-pairs[:,1]) > 1
    for dsb_pair in input_params.dsb_indices:
        for n,p in enumerate(pairs):
            if p[0] == dsb_pair[0] and p[1] == dsb_pair[1]:
                mask[n] = False
                break
    pairs = pairs[mask]
    # get distances of residue pairs
    d = md.compute_distances(traj, pairs)
    if frame_list is not None:
        d = d[frame_list]
    # calculate interaction parameters for residue pairs
    residues_0 = [input_params.seq_list[i] for i in pairs[:,0]]
    residues_1 = [input_params.seq_list[i] for i in pairs[:,1]]
    sigma_values_0 = res_params.sigmas.loc[residues_0].values
    sigma_values_1 = res_params.sigmas.loc[residues_1].values
    sigmas = (sigma_values_0 + sigma_values_1)/2
    lambda_values_0 = res_params.lambdas.loc[residues_0].values
    lambda_values_1 = res_params.lambdas.loc[residues_1].values
    lambdas = (lambda_values_0 + lambda_values_1)/2
    q_values_0 = res_params.q.loc[residues_0].values
    q_values_1 = res_params.q.loc[residues_1].values
    qq = q_values_0*q_values_1
    # calculate Ashbaugh-Hatch energies
    AH_SR = lambda r,s,l : 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)
    AH_LR = lambda r,s,l : 4*0.8368*l*((s/r)**12-(s/r)**6)
    AH = lambda r,s,l : np.where(r<2**(1/6)*s, AH_SR(r,s,l), AH_LR(r,s,l))
    AH_trunc = lambda r,s,l,rc : np.where(r<rc, AH(r,s,l)-AH(rc,s,l), 0)
    AH_frames = AH_trunc(d, sigmas[np.newaxis,:], lambdas[np.newaxis,:], 2.0)
    AH_energies = np.nansum(AH_frames, axis=0)/AH_frames.shape[0]
    # calculate Debye-Huckel energies
    kT = 8.3145*input_params.T*1e-3
    f = lambda T: 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = f(input_params.T)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    kappa = np.sqrt(8*np.pi*lB*input_params.I*6.022/10)
    DH = lambda r,qq: qq*lB*kT*np.exp(-kappa*r)/r
    DH_trunc = lambda r,qq,rc: np.where(r<rc, DH(r,qq)-DH(rc,qq), 0)
    DH_frames = DH_trunc(d, qq[np.newaxis,:], 4.0)
    DH_energies = np.nansum(DH_frames, axis=0)/DH_frames.shape[0]
    # get lists to account for padding and masking
    if pad_mask is None:
        map_structure = range(n_res)
        profile_mask = [1]*n_res
    else:
        map_structure, profile_mask = get_pad_mask(pad_mask)
    map_dim = len(map_structure)
    map_positions = []
    for i in range(n_res):
        to_add = [j for j, element in enumerate(map_structure) if element == i]
        map_positions.append(to_add)
    # create maps
    sum_map = pd.DataFrame(index=range(map_dim), columns=range(map_dim),
                           dtype=float)
    AH_map = pd.DataFrame(index=range(map_dim), columns=range(map_dim),
                          dtype=float)
    DH_map = pd.DataFrame(index=range(map_dim), columns=range(map_dim),
                          dtype=float)
    for k,(i,j) in enumerate(pairs):
        for i_pos in map_positions[i]:
            for j_pos in map_positions[j]:
                sum_map.loc[i_pos,j_pos] = AH_energies[k] + DH_energies[k]
                sum_map.loc[j_pos,i_pos] = AH_energies[k] + DH_energies[k]
                AH_map.loc[i_pos,j_pos] = AH_energies[k]
                AH_map.loc[j_pos,i_pos] = AH_energies[k]
                DH_map.loc[i_pos,j_pos] = DH_energies[k]
                DH_map.loc[j_pos,i_pos] = DH_energies[k]
    # create energy profiles
    mask_list = [i for i, keep in enumerate(profile_mask) if not keep]
    sum_map_masked = sum_map.copy(deep=True)
    AH_map_masked = AH_map.copy(deep=True)
    DH_map_masked = DH_map.copy(deep=True)
    sum_map_masked.loc[:,mask_list] = np.nan
    AH_map_masked.loc[:,mask_list] = np.nan
    DH_map_masked.loc[:,mask_list] = np.nan
    sum_profile_array = np.array(sum_map_masked.sum(axis=1))
    AH_profile_array = np.array(AH_map_masked.sum(axis=1))
    DH_profile_array = np.array(DH_map_masked.sum(axis=1))
    sum_profile_array[mask_list] = np.nan
    AH_profile_array[mask_list] = np.nan
    DH_profile_array[mask_list] = np.nan
    sum_profile = ArrayAverage(sum_profile_array)
    AH_profile = ArrayAverage(AH_profile_array)
    DH_profile = ArrayAverage(DH_profile_array)
    # package and return variables
    sum_map_profile = MapProfile(sum_map, sum_profile)
    AH_map_profile = MapProfile(AH_map, AH_profile)
    DH_map_profile = MapProfile(DH_map, DH_profile)
    return sum_map_profile, AH_map_profile, DH_map_profile

# function to create contact and energy plots
def contact_and_energy(input_params, res_params, proj_dir,
                       plot_width, plot_height, save=False, show=False,
                       frame_list=None, pad_mask=None,
                       d_threshold=0.8, contact_limit=1e-3):

    ### LOAD TRAJECTORY AND ANALYSE ###
    traj = md.load_dcd(proj_dir + '/SIMULATION/traj.dcd',
                       top = proj_dir + '/SIMULATION/top.pdb')
    contacts = analyse_contacts(traj, input_params, res_params,
                                frame_list=frame_list, pad_mask=pad_mask,
                                d_threshold=d_threshold)
    sum_energy, AH_energy, DH_energy = analyse_energies(traj, input_params,
                                                        res_params,
                                                        frame_list=frame_list,
                                                        pad_mask=pad_mask)

    ### CREATE PLOTS ###
    mpl.rcParams.update({'font.size': 7})
    fig, axs = plt.subplots(2, 4, figsize=(plot_width, plot_height),
                            facecolor='w', dpi=300, layout='constrained')
    axs = axs.flatten()
    # axis labels
    residue_label = r'Residue #'
    contact_map_label = r'Interaction probability'
    energy_map_label = r'Energy  /  J mol$^{-1}$'
    contact_profile_label = r'Total non-bonded contacts'
    energy_profile_label = r'Total energy  /  kJ mol$^{-1}$'
    # plot contact map and profile
    contacts.plot_contact_map(axs[0], residue_label, contact_map_label,
                              contact_limit=contact_limit)
    contacts.plot_contact_profile(axs[4], residue_label, contact_profile_label)
    # energy maps and profiles to loop through
    objects = [sum_energy, AH_energy, DH_energy]
    # iterate through variables simultaneously
    for i, obj in enumerate(objects, 1):
        obj.plot_energy_map(axs[i], residue_label, energy_map_label)
        obj.plot_energy_profile(axs[i+4], residue_label, energy_profile_label)
    # create dataframes of contact/energy profiles to export
    contact_profile = np.c_[contacts.profile.array,
                            contacts.profile.running_avg()]
    sum_profile = np.c_[sum_energy.profile.array,
                        sum_energy.profile.running_avg()]
    AH_profile = np.c_[AH_energy.profile.array,
                       AH_energy.profile.running_avg()]
    DH_profile = np.c_[DH_energy.profile.array,
                       DH_energy.profile.running_avg()]
    contact_profile_cols = ['Contact probability', 'running_average(5)']
    energy_profile_cols = ['Energy / J mol$^{-1}$', 'running_average(5)']
    contact_profile_df = pd.DataFrame(data=contact_profile,
                                      columns=contact_profile_cols)
    sum_profile_df = pd.DataFrame(data=sum_profile, columns=energy_profile_cols)
    AH_profile_df = pd.DataFrame(data=AH_profile, columns=energy_profile_cols)
    DH_profile_df = pd.DataFrame(data=DH_profile, columns=energy_profile_cols)

    ### SAVE DATA IF REQUESTED ###
    if save:
        dir = proj_dir + '/contacts_and_energy'
        try:
            os.mkdir(dir)
        except:
            pass
        plt.savefig(dir + '/contacts_and_energy.pdf', dpi=300, facecolor='w',
                    edgecolor='w', orientation='portrait', bbox_inches='tight')
        contacts.map.to_csv(dir + '/contact_map.csv')
        sum_energy.map.to_csv(dir + '/sum_energy_map.csv')
        AH_energy.map.to_csv(dir + '/non_charged_energy_map.csv')
        DH_energy.map.to_csv(dir + '/charged_energy_map.csv')
        contact_profile_df.to_csv(dir + '/contact_profile.csv')
        sum_profile_df.to_csv(dir + '/sum_energy_profile.csv')
        AH_profile_df.to_csv(dir + '/non_charged_energy_profile.csv')
        DH_profile_df.to_csv(dir + '/charged_energy_profile.csv')
        print('Saved contact and energy plots.')

    ### SHOW PLOTS IF REQUESTED ###
    if show:
        plt.show()
    else:
        plt.close()
