#@title <b><font color='#19b319'>Initial setup (iii):</b> simulation toolbox


### SECTION 1 ###
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


### SECTION 2 ###
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


### SECTION 3 ###
### SIMULATION-ASSOCIATED CODE TO HANDLE DATA OUTPUT ###

# show progress bar
def progress(value, max=100):
    return HTML('''
        <progress
            value='{value}'
            max='{max}',
            style='width: 25%'
        >
            {value}
        </progress>
    '''.format(value=value, max=max))

# generate coordinates and trajectory in convenient formats
def gen_DCD(name, sim_dir, eq_frames=10):
    traj = md.load(sim_dir + '/pretraj.dcd', top = sim_dir + '/top.pdb')
    traj = traj.image_molecules(inplace=False,
                                anchor_molecules=[set(traj.top.chain(0).atoms)],
                                make_whole=True)
    traj.center_coordinates()
    traj.xyz += traj.unitcell_lengths[0, 0]/2
    traj[eq_frames:].save_dcd(sim_dir + '/traj.dcd')


### SECTION 4 ###
### MAIN SIMULATION CODE ###

# main function to run simulation
def simulate(input_params, res_params, sim_dir,
             stride=7000, eq_frames=10, n_frames=5000):

    ### INITIAL SETUP ###
    # create system object
    system = openmm.System()

    ### SET NUMBER OF RESIDUES AND CREATE BOX ###
    N = len(input_params.sequence)
    L = (N-1)*0.76+4
    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = L * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    ### SET SYSTEM TOPOLOGY AND INITIAL STATE ###
    # create topology object
    top = md.Topology()
    chain = top.add_chain() # create single chain
    # create initial coordinates
    pos = [[0,0,L/2+(i-N/2.)*.38] for i in range(N)]
    # add residues to the topology
    for res in input_params.seq_list:
        residue = top.add_residue(res, chain)
        top.add_atom(res, element=md.element.carbon, residue=residue)
    # create bonds along the chain
    for i in range(chain.n_atoms-1):
        top.add_bond(chain.atom(i),chain.atom(i+1))
    # save topology file
    tr = md.Trajectory(np.array(pos).reshape(N,3), top, 0, [L,L,L], [90,90,90])
    tr.save_pdb(sim_dir + '/top.pdb')
    # store topology as pdb
    pdb = app.pdbfile.PDBFile(sim_dir + '/top.pdb')
    # add particles to system
    for a in input_params.seq_list:
        system.addParticle(res_params.loc[a].MW*unit.amu)

    ### SET UP FORCE FIELD ###
    # main chain bonds, harmonic potential
    hb = openmm.openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)
    # disulfide bonds, harmonic potential
    dsb = openmm.openmm.HarmonicBondForce()
    dsb.setUsesPeriodicBoundaryConditions(True)
    # electrostatics, Yukawa potential
    dh = DH_Params(input_params, res_params) # get Debye-Huckel parameters
    yu_expression = 'q*(exp(-kappa*r)/r - exp(-kappa*4)/4); q=q1*q2'
    yu = openmm.openmm.CustomNonbondedForce(yu_expression) # create force
    yu.addGlobalParameter('kappa', dh.kappa/unit.nanometer)
    yu.addPerParticleParameter('q')
    for e in dh.epsilon:
        yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
    yu.setForceGroup(0)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    yu.setCutoffDistance(4*unit.nanometer)
    # stickiness, Ashbaugh-Hatch potential
    lj = LJ_Params(input_params, res_params) # get Lennard-Jones parameters
    ah_expression = ('select(step(r-2^(1/6)*s),' +
        '4*eps*l*((s/r)^12-(s/r)^6-shift),' +
        '4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l));' +
        's=0.5*(s1+s2); l=0.5*(l1+l2);' +
        'shift=(0.5*(s1+s2)/2)^12-(0.5*(s1+s2)/2)^6')
    ah = openmm.openmm.CustomNonbondedForce(ah_expression) # create force
    ah.addGlobalParameter('eps', lj.epsilon*unit.kilojoules_per_mole)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    for a in input_params.seq_list:
        ah.addParticle([res_params.loc[a].sigmas*unit.nanometer,
                        res_params.loc[a].lambdas*unit.dimensionless])
    ah.setForceGroup(1)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(2*unit.nanometer)
    # create main chain bonds and exclusions
    for i in range(N-1):
        hb_distance = 0.38*unit.nanometer
        hb_spring_const = 8033*unit.kilojoules_per_mole/(unit.nanometer**2)
        hb.addBond(i, i+1, hb_distance, hb_spring_const)
        yu.addExclusion(i, i+1)
        ah.addExclusion(i, i+1)
    # create disulfide bonds and exclusions
    for i in dsb_indices:
        dsb_distance = 0.56*unit.nanometer
        dsb_spring_const = 850*unit.kilojoules_per_mole/(unit.nanometer**2)
        hb.addBond(i[0], i[1], dsb_distance, dsb_spring_const)
        yu.addExclusion(i[0], i[1])
        ah.addExclusion(i[0], i[1])
    # add forces to system
    system.addForce(hb)
    system.addForce(dsb)
    system.addForce(yu)
    system.addForce(ah)

    ### SET UP SIMULATION ###
    # integrator parameters
    temp = input_params.T*unit.kelvin
    friction = 0.01/unit.picosecond
    timestep = 0.010*unit.picosecond #10 fs timestep
    # set up integrator
    integrator = openmm.openmm.LangevinIntegrator(temp, friction, timestep)
    # set up platform
    platform = openmm.Platform.getPlatformByName('CUDA')
    # set up simulation
    simulation = app.simulation.Simulation(pdb.topology, system, integrator,
                                        platform, dict(CudaPrecision='mixed'))
    # either load check point file, or minimise energy from initial coords
    checkpoint = sim_dir + '/restart.chk'
    if os.path.isfile(checkpoint):
        print('Reading check point file')
        simulation.loadCheckpoint(checkpoint)
        simulation.reporters.append(
            app.dcdreporter.DCDReporter(sim_dir + '/pretraj.dcd',
                                        stride, append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(
            app.dcdreporter.DCDReporter(sim_dir + '/pretraj.dcd', stride))
    # add traj file as reporter
    simulation.reporters.append(app.statedatareporter.StateDataReporter(
        sim_dir + '/traj.log', stride,
        potentialEnergy=True, temperature=True, step=True, speed=True,
        elapsedTime=True, separator='\t'))

    ### RUN SIMULATION AND GENERATE TRAJECTORY ###
    # calculate number of simulation steps
    n_steps = stride*(eq_frames + n_frames)
    # split simulation steps into blocks of 10000 frames
    block_size = 10000
    n_complete_blocks = n_steps // block_size
    last_block_size = n_steps - (block_size*n_complete_blocks)
    # create progress bar, which will be updated at the end of each block
    print('Simulating...')
    simulation_progress = display(progress(0, n_steps), display_id=True)
    # iterate through blocks, updating progress periodically
    for n in range(0, n_complete_blocks):
        simulation.step(block_size)
        simulation_progress.update(progress(block_size*(n+1), n_steps))
    # last block, which may be smaller than the pthers
    simulation.step(last_block_size)
    simulation_progress.update(progress(n_steps, n_steps))
    # save and create DCD
    simulation.saveCheckpoint(checkpoint)
    gen_DCD(input_params.seq_name, sim_dir, eq_frames)
    print('Done.')

#@title <b><font color='#19b319'>Initial setup (iv):</b> analysis toolbox

### SECTION 1 ###
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


### SECTION 2 ###
### CLASSES AND FUNCTIONS FOR SIMULATION ANALYSIS ###

# get all the factors of a number; used to work out block sizes
def get_factors(x):
    factors = []
    for i in range(1, int(np.sqrt(x)) + 1):
        if x % i == 0:
            factors.append(i)
            if i != x // i:
                factors.append(x // i)
    factors.sort()
    return factors

# estimate the error of correlated variables using the blocking algorithm
def blocking(array):
    # get different block sizes to split the array into
    dimension = len(array)
    n_blocks = get_factors(dimension)[1:] # exclude unblocked case
    block_sizes = [dimension // n for n in n_blocks]
    # create a list of errors and errors-of-errors for different block sizes
    array_mean = array.mean()
    error_list = []
    error_of_error_list = []
    for n, bs in zip(n_blocks, block_sizes):
        averages = np.array([array[bs*i : bs*(i+1)].mean() for i in range(n)])
        error = np.sqrt(np.square(averages - array_mean).mean() / (n-1))
        error_list.append(error)
        error_of_error = error / np.sqrt(2*(n-1))
        error_of_error_list.append(error_of_error)
    '''
    Maximising this function (achieved below by minimising its negative)
    gives a metric that defines how well-converged the error estimates are
    at a particular block size, calculated as the number of larger blocks
    whose error ranges overlap with a fitted value constrained to lie within
    the error range of the current block.
    '''
    def block_convergence(x, larger_blocks):
        n_overlap = 0
        for lb in larger_blocks:
            if (x >= lb[1]-lb[2]) and (x <= lb[1]+lb[2]):
                n_overlap += 1
        return n_overlap
    '''
    For each block size, get the degree of convergence by fitting the
    block_convergence function, ie. fitting a value constrained to lie within
    the block's own error bounds so that it also lies within the error bounds
    of as many of the larger blocks as possible.
    '''
    convergence_list = [] # list of convergence values by block size
    to_iterate = np.flip(np.c_[block_sizes, error_list, error_of_error_list],
                         axis=0) # will iterate from smaller to larger
    for block_info in to_iterate:
        size = block_info[0]
        error = block_info[1]
        error_of_error = [2]
        # get list of blocks larger than this one
        args = np.array([i for i in to_iterate if i[0]>size])
        # calculate bounds
        lower_bound = error - error_of_error
        upper_bound = error + error_of_error
        bounds = [(lower_bound, upper_bound)]
        # maximum possible overlap between this block's error range and others
        n_overlap = -minimize(fun=lambda x,lb: -block_convergence(x, lb),
                              x0=error, args=args, bounds=bounds).fun
        convergence_list.append(n_overlap)
    # return the block error that maximises the convergence criterion
    block_size = to_iterate[np.argmax(convergence_list), 0]
    est_error = to_iterate[np.argmax(convergence_list), 1]
    return block_size, est_error

# just get the block size
def block_size(array):
    block_size, error = blocking(array)
    return block_size

# just get the blocking error
def blocking_error(array):
    block_size, error = blocking(array)
    return error

# to store arrays and their summary stats
class ArrayAverage:
    def __init__(self, array, use_blocking=False):
        self.array = array.copy() # 1D array of data
        self.value = self.array.mean()
        self.sigma = self.array.std()
        self.count = len(self.array)
        if use_blocking:
            last_axis = len(array.shape)-1
            self.error = np.apply_along_axis(blocking_error, last_axis, array)
        else:
            self.error = self.sigma / np.sqrt(self.count)
    # calculate a frequency distribution using a kernel density estimation (KDE)
    def KDE(self):
        x = np.linspace(np.min(self.array), np.max(self.array), num=100)
        y = gaussian_kde(self.array, bw_method='silverman').evaluate(x)
        y /= np.sum(y)
        return x, y
    # get summary statistics as an array of strings - for printing purposes
    def get_str_array(self):
        value_string = '{:.4f}'.format(self.value)
        error_string = '{:.4f}'.format(self.error)
        sigma_string = '{:.4f}'.format(self.sigma)
        return [value_string, error_string, sigma_string]
    # calculate running average, window = 2*either_side + 1
    def running_avg(self, either_side=2):
        ra = []
        for n in range(0, len(self.array)-(2*either_side)):
            ra.append(np.sum(self.array[n:n+(2*either_side)+1]))
        padding = [np.NAN]*either_side
        return np.array(padding + ra + padding)/((2*either_side)+1)
    # plot the KDE, mean, and standard error
    def create_plot(self, ax, x_name, y_name):
        # plot KDE estimate of frequency distribution
        x, y = self.KDE()
        ax.plot(x, y)
        # set axis labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        # determine upper limit of y axis
        y_upper = 1.1*y.max()
        ax.set_ylim(0, y_upper)
        # plot mean and standard error
        ax.vlines(self.value, 0, y_upper)
        range_lower = self.value - self.error
        range_upper = self.value + self.error
        ax.fill_between([range_lower, range_upper], 0, y_upper, alpha=0.3)
        # needs minor ticks!
        ax.minorticks_on()

# to store best-fit quantities and their errors
class FittedQuantity:
    def __init__(self, best_fit, error):
        self.value = best_fit
        self.error = error
    # get summary statistics as an array of strings - for printing purposes
    def get_str_array(self):
        value_string = '{:.4f}'.format(self.value)
        error_string = '{:.4f}'.format(self.error)
        sigma_string = ''
        return [value_string, error_string, sigma_string]

# to store Flory scaling exponent data
class FloryData:
    def __init__(self, ij, dij, R0, nu):
        self.ij = ij
        self.dij = dij
        self.R0 = R0
        self.nu = nu
    # plot the fit used to determine the Flory exponent
    def create_plot(self, ax, x_name, y_name, fontsize=7):
        # plot the distances versus sequence separation
        ax.plot(self.ij, self.dij)
        # plot the fit
        dij_fit = self.R0.value*np.power(self.ij, self.nu.value)
        ax.plot(self.ij, self.R0.value*np.power(self.ij, self.nu.value),
                c='0.3',ls='dashed',label='Fit')
        # set axis labels
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        # add the Flory exponent as text
        ax.text(0.05, 0.9, r'$\nu$={:.2f}'.format(self.nu.value),
                horizontalalignment='left',verticalalignment='center',
                transform=ax.transAxes, fontsize=fontsize)
        # add legend
        ax.legend(loc='lower right')
        # needs minor ticks!
        ax.minorticks_on()

# to store and plot contact/energy maps and profiles
class MapProfile:
    def __init__(self, map, profile):
        self.map = map
        self.profile = profile
    def plot_contact_map(self, ax, xy_label, v_label, contact_limit=1e-3):
        # make a copy to display, and truncate to fit axes
        to_display = self.map.copy()
        vmin = contact_limit
        vmax = 1.0
        to_display = np.clip(to_display, vmin, vmax)
        # plot contact map and colour bar
        im = ax.imshow(to_display,
                       extent=[1, to_display.shape[0], 1, to_display.shape[0]],
                       origin='lower', aspect='equal', cmap=plt.cm.Blues,
                       norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        c_bar = plt.colorbar(im, ax=ax, label=v_label, fraction=0.05, pad=0.04)
        # set axis labels
        ax.set_xlabel(xy_label)
        ax.set_ylabel(xy_label)
    def plot_contact_profile(self, ax, x_label, y_label):
        # get residue indices
        res_indices = np.array(range(1, len(self.profile.array)+1))
        # set  axis ranges
        nan_removed = self.profile.array[~np.isnan(self.profile.array)]
        ax.set_xlim(res_indices[0]-1, res_indices[-1]+1)
        ax.set_ylim(0, 1.1*nan_removed.max())
        # plot raw contact profile as points
        ax.scatter(res_indices, self.profile.array, s=10, marker='x')
        # plot running average as a line
        ax.plot(res_indices, self.profile.running_avg(), color='limegreen',
                linewidth=1.2)
        # set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # needs minor ticks!
        ax.minorticks_on()
    def plot_energy_map(self, ax, xy_label, v_label):
        # convert from kJ/mol to J/mol
        to_display = 1e3*self.map
        # truncate to fit axes
        vmin = -1000
        vmax = 1000
        to_display[to_display < vmin] = vmin
        to_display[to_display > vmax] = vmax
        # plot energy map and colour bar
        im = ax.imshow(to_display,
                       extent=[1, to_display.shape[0], 1, to_display.shape[0]],
                       origin='lower', aspect='equal', cmap=plt.cm.RdBu_r,
                       norm=mpl.colors.SymLogNorm(linthresh=1,
                                                  vmin=vmin, vmax=vmax))
        c_bar = plt.colorbar(im, ax=ax, label=v_label, fraction=0.05, pad=0.04)
        # set axis labels
        ax.set_xlabel(xy_label)
        ax.set_ylabel(xy_label)
    def plot_energy_profile(self, ax, x_label, y_label):
        # cache variables; do not convert units
        ep = self.profile.array
        ra = self.profile.running_avg()
        # get residue indices
        res_indices = np.array(range(1, len(ep)+1))
        # set  axis ranges
        y_min = ep[~np.isnan(ep)].min()
        y_max = ep[~np.isnan(ep)].max()
        y_range = y_max-y_min
        padding = 0.1*y_range
        ax.set_xlim(res_indices[0]-1, res_indices[-1]+1)
        ax.set_ylim(min(y_min-padding, 0), max(y_max+padding, 0))
        # add horizontal line at zero
        ax.axhline(0, color='tab:gray', linewidth=1.2)
        # plot raw energy profile as points
        ax.scatter(res_indices, ep, s=10, marker='x')
        # plot running average as a line
        ax.plot(res_indices, ra, color='limegreen', linewidth=1.2)
        # set axis labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # needs minor ticks!
        ax.minorticks_on()

### SECTION 3 ###
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


### SECTION 4 ###
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
