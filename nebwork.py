from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Str, Float, Bool, List
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.folder import FolderData
from aiida.orm.code import Code
from aiida.common.exceptions import NotExistent
# from aiida.orm.data.structure import StructureData

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation


class NEBWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(NEBWorkchain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("struc_folder", valid_type=FolderData)
        spec.input("wfn_cp_commands", valid_type=List)
        spec.input("num_machines", valid_type=Int)
        spec.input("calc_name", valid_type=Str)
        spec.input("cell", valid_type=Str)
        spec.input("fixed_atoms", valid_type=Str)

        spec.input("nproc_rep", valid_type=Int)
        spec.input("nreplicas", valid_type=Int)
        spec.input("spring", valid_type=Float)
        spec.input("rotate", valid_type=Bool)
        spec.input("align", valid_type=Bool)
        spec.input("nstepsit", valid_type=Int)
        spec.input("endpoints", valid_type=Bool)
        spec.input("calc_type", valid_type=Str)
        spec.input("first_slab_atom", valid_type=Int)
        spec.input("last_slab_atom", valid_type=Int)

        spec.outline(
            cls.init,
            cls.calc_neb,
            #while_(cls.not_converged)(
            #    cls.calc_neb
            #),
            # cls.store_neb
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        try:
            self.report('Convergence check DEBUG: {}'.format(self.ctx.neb))
            self.report('Convergence check: {}'
                        .format(self.ctx.neb.res.exceeded_walltime))
            return self.ctx.neb.res.exceeded_walltime
        except AttributeError:
            return True
        except NotExistent:
            return False

    # ==========================================================================
    def init(self):
        self.report('Init NEB')
        # Set the restart folder
        try:
            self.ctx.remote_calc_folder = self.ctx.neb.remote_calc_folder
        except AttributeError:
            self.ctx.remote_calc_folder = None

        # Here we need to create the xyz files of all the replicas
        self.ctx.this_name = self.inputs.calc_name
        self.ctx.file_list = self.inputs.struc_folder.get_folder_list()
        self.ctx.n_files = len(self.ctx.file_list)-2

        # Report some things
        self.report('Passed #{} replica geometries'.format(self.ctx.n_files))
        self.report('Replicas: {}'.format(self.ctx.file_list))

    # ==========================================================================
    def calc_neb(self):
        self.report("Running CP2K CI-NEB calculation."
                    .format(self.ctx.this_name))

        inputs = self.build_calc_inputs(self.inputs.struc_folder,
                                        # Setup calculation
                                        self.inputs.cell,
                                        self.inputs.cp2k_code,
                                        self.inputs.fixed_atoms,
                                        self.inputs.num_machines,
                                        self.ctx.remote_calc_folder,
                                        self.inputs.wfn_cp_commands,
                                        # NEB input
                                        self.inputs.align,
                                        self.inputs.endpoints,
                                        self.inputs.nproc_rep,
                                        self.inputs.nreplicas,
                                        self.inputs.nstepsit,
                                        self.inputs.rotate,
                                        self.inputs.spring,
                                        # Calculation type specific
                                        self.inputs.calc_type,
                                        self.ctx.file_list,
                                        # find this in the workflow
                                        # instead of passing
                                        self.inputs.first_slab_atom,
                                        self.inputs.last_slab_atom)

        self.report(" ")
        self.report("inputs: "+str(inputs))
        self.report(" ")
        future = submit(Cp2kCalculation.process(), **inputs)
        self.report("future: "+str(future))
        self.report(" ")
        return ToContext(neb=Calc(future))

    # ==========================================================================
    def store_replica(self):
        return self.out('replica_{}'.format(self.ctx.this_name),
                        self.ctx.neb.out.output_structure)

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, struc_folder, cell, code,
                          fixed_atoms, num_machines, remote_calc_folder,
                          wfn_cp_commands,
                          # NEB input
                          align, endpoints, nproc_rep, nreplicas, nstepsit,
                          rotate, spring,
                          #list of available wfn
                          # Calculation type specific
                          calc_type, file_list, first_slab_atom,
                          last_slab_atom):

        inputs = {}
        inputs['_label'] = "NEB"

        inputs['code'] = code
        inputs['file'] = {}

        # The files passed by the notebook
        for f in struc_folder.get_folder_list():
            path = struc_folder.get_abs_path()+'/path/'+f
            inputs['file'][f] = SinglefileData(file=path)

        # Au potential
        pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
        inputs['file']['au_pot'] = pot_f

        remote_computer = code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        if 'Mixed' in str(calc_type):
            # Then we have mol0.xyz which is not a replica itself
            nreplica_files = len(file_list)-1
        else:
            nreplica_files = len(file_list)

        if calc_type == 'Mixed DFTB':
            walltime = 18000
        else:
            walltime = 86000

        inp = cls.get_cp2k_input(cell=cell,
                                 fixed_atoms=fixed_atoms,
                                 machine_cores=machine_cores*num_machines,
                                 # NEB input
                                 align=align,
                                 endpoints=endpoints,
                                 nproc_rep=nproc_rep,
                                 nreplicas=nreplicas,
                                 nstepsit=nstepsit,
                                 rotate=rotate,
                                 spring=spring,
                                 # Calculation specific
                                 calc_type=calc_type,
                                 nreplica_files=nreplica_files,
                                 first_slab_atom=first_slab_atom,
                                 last_slab_atom=last_slab_atom,
                                 walltime=walltime*0.97)

        if remote_calc_folder is not None:
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.xyz',
                                                                    '*.out',
                                                                    '*.ener']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }
        if len(wfn_cp_commands) > 0:
            inputs['_options']["prepend_text"] = ""
            for wfn_cp_command in wfn_cp_commands:
                inputs['_options']["prepend_text"] += wfn_cp_command + "\n"
        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell, fixed_atoms, machine_cores,
                       align, endpoints, nproc_rep, nreplicas,
                       nstepsit, rotate, spring,
                       calc_type, nreplica_files,
                       first_slab_atom, last_slab_atom,
                       walltime):
        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'BAND',
                'WALLTIME': walltime,
                'PRINT_LEVEL': 'LOW'
            },
            'MOTION': cls.get_motion(align, endpoints, fixed_atoms, nproc_rep,
                                     nreplicas, nstepsit, rotate, spring,
                                     nreplica_files),
            'FORCE_EVAL': [],
        }

        if calc_type == 'Mixed DFTB':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores),
                                 cls.force_eval_fist(cell),
                                 cls.get_force_eval_qs_dftb(cell)]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Mixed DFT':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores),
                                 cls.force_eval_fist(cell),
                                 cls.get_force_eval_qs_dft(cell,
                                                           'mol0.xyz')]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Full DFT':
            # The right structure!
            inp['FORCE_EVAL'] = [cls.get_force_eval_qs_dft(cell,
                                                           'replica1.xyz')]
        return inp

    # ==========================================================================
    @classmethod
    def force_eval_mixed(cls, cell, first_slab_atom, last_slab_atom,
                         machine_cores):
        first_mol_atom = 1
        last_mol_atom = first_slab_atom - 1

        mol_delim = (first_mol_atom, last_mol_atom)
        slab_delim = (first_slab_atom, last_slab_atom)

        force_eval = {
            'METHOD': 'MIXED',
            'MIXED': {
                'MIXING_TYPE': 'GENMIX',
                'GROUP_PARTITION': '2 %d' % (machine_cores-2),
                'GENERIC': {
                    'ERROR_LIMIT': '1.0E-10',
                    'MIXING_FUNCTION': 'E1+E2',
                    'VARIABLES': 'E1 E2'
                },
                'MAPPING': {
                    'FORCE_EVAL_MIXED': {
                        'FRAGMENT':
                            [{'_': '1', ' ': '%d  %d' % mol_delim},
                             {'_': '2', ' ': '%d  %d' % slab_delim}],
                    },
                    'FORCE_EVAL': [{'_': '1', 'DEFINE_FRAGMENTS': '1 2'},
                                   {'_': '2', 'DEFINE_FRAGMENTS': '1'}],
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'replica1.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            }
        }

        return force_eval

    # ==========================================================================
    @classmethod
    def force_eval_fist(cls, cell):
        ff = {
            'SPLINE': {
                'EPS_SPLINE': '1.30E-5',
                'EMAX_SPLINE': '0.8',
            },
            'CHARGE': [],
            'NONBONDED': {
                'GENPOT': [],
                'LENNARD-JONES': [],
                'EAM': {
                    'ATOMS': 'Au Au',
                    'PARM_FILE_NAME': 'Au.pot',
                },
            },
        }

        for x in ('Au', 'H', 'C', 'O', 'N'):
            ff['CHARGE'].append({'ATOM': x, 'CHARGE': 0.0})

        genpot_fun = 'A*exp(-av*r)+B*exp(-ac*r)-C/(r^6)/( 1+exp(-20*(r/R-1)) )'
        genpot_val = '4.13643 1.33747 115.82004 2.206825'\
                     ' 113.96850410723008483218 5.84114'
        for x in ('C', 'N', 'O', 'H'):
            ff['NONBONDED']['GENPOT'].append(
                {'ATOMS': 'Au ' + x,
                 'FUNCTION': genpot_fun,
                 'VARIABLES': 'r',
                 'PARAMETERS': 'A av B ac C R',
                 'VALUES': genpot_val,
                 'RCUT': '15'}
            )

        for x in ('C H', 'H H', 'H N', 'C C', 'C O', 'C N', 'N N', 'O H',
                  'O N', 'O O'):
            ff['NONBONDED']['LENNARD-JONES'].append(
                {'ATOMS': x,
                 'EPSILON': '0.0',
                 'SIGMA': '3.166',
                 'RCUT': '15'}
            )

        force_eval = {
            'METHOD': 'FIST',
            'MM': {
                'FORCEFIELD': ff,
                'POISSON': {
                    'EWALD': {
                      'EWALD_TYPE': 'none',
                    },
                },
            },
            'SUBSYS': {
                'CELL': {
                    'ABC': cell,
                },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'replica1.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            },
        }
        return force_eval

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dftb(cls, cell):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'QS': {
                    'METHOD': 'DFTB',
                    'EXTRAPOLATION': 'USE_GUESS',
                    'DFTB': {
                        'SELF_CONSISTENT': 'T',
                        'DISPERSION': 'T',
                        'ORTHOGONAL_BASIS': 'F',
                        'DO_EWALD': 'F',
                        'PARAMETER': {
                            'PARAM_FILE_PATH': 'DFTB/scc',
                            'PARAM_FILE_NAME': 'scc_parameter',
                            'UFF_FORCE_FIELD': '../uff_table',
                        },
                    },
                },
                'SCF': {
                    'MAX_SCF': '30',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-6',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '20',
                        'EPS_SCF': '1.0E-6',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol0.xyz',
                    'COORDINATE': 'xyz'
                }
            }
        }

        return force_eval

    # ==========================================================================
    @classmethod
    def get_motion(cls, align, endpoints, fixed_atoms, nproc_rep, nreplicas,
                   nstepsit, rotate, spring, nreplica_files):
        motion = {
            'CONSTRAINT': {
                'FIXED_ATOMS': {
                    'LIST': '{}'.format(fixed_atoms)
                }
            },
            'BAND': {
                'NPROC_REP': nproc_rep,
                'BAND_TYPE': 'CI-NEB',
                'NUMBER_OF_REPLICA': nreplicas,
                'K_SPRING': str(spring),
                'CONVERGENCE_CONTROL': {
                    'MAX_FORCE': '0.0005',
                    'RMS_FORCE': '0.001',
                    'MAX_DR': '0.002',
                    'RMS_DR': '0.005'
                },
                'ROTATE_FRAMES': str(rotate)[0],
                'ALIGN_FRAMES': str(align)[0],
                'CI_NEB': {
                    'NSTEPS_IT': str(nstepsit)
                },
                'OPTIMIZE_BAND': {
                    'OPT_TYPE': 'DIIS',
                    'OPTIMIZE_END_POINTS': str(endpoints)[0],
                    'DIIS': {
                        'MAX_STEPS': 1000
                    }
                },
                'PROGRAM_RUN_INFO': {
                    'INITIAL_CONFIGURATION_INFO': ''
                },
                'CONVERGENCE_INFO': {
                    '_': ''
                },
                'REPLICA': []
            },
        }

        # The fun part
        for r in range(nreplica_files):
            motion['BAND']['REPLICA'].append({
                'COORD_FILE_NAME': 'replica{}.xyz'.format(r+1)
            })

        return motion

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell, coord_file_name):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                #'RESTART_FILE_NAME': './parent_calc/aiida-RESTART.wfn',
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'USE_GUESS',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '600',
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '30',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-6',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '50',
                        'EPS_SCF': '1.0E-6',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                },
                'XC': {
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                    'VDW_POTENTIAL': {
                        'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                        'PAIR_POTENTIAL': {
                            'TYPE': 'DFTD3',
                            'CALCULATE_C9_TERM': '.TRUE.',
                            'PARAMETER_FILE_NAME': 'dftd3.dat',
                            'REFERENCE_FUNCTIONAL': 'PBE',
                            'R_CUTOFF': '[angstrom] 15',
                        }
                    }
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell},
                'TOPOLOGY': {
                    # starting geometry
                    'COORD_FILE_NAME': coord_file_name,
                    'COORDINATE': 'xyz',
                },
                'KIND': [],
            }
        }

        force_eval['SUBSYS']['KIND'].append({
            '_': 'Au',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'C',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q4'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Br',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'O',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q6'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'N',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q5'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'I',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'H',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q1'
        })

        return force_eval
