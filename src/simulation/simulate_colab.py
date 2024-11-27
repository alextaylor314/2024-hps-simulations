import os
import numpy as np
import mdtraj as md
import openmm
from openmm import app
from simtk import unit
from IPython.display import display, HTML


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
