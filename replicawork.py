from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Str, Float
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.remote import RemoteData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil

import numpy as np


class ReplicaWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(ReplicaWorkchain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("num_machines", valid_type=Int, default=Int(54))
        spec.input("replica_name", valid_type=Str)
        spec.input("cell", valid_type=Str, default=Str(''))
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
        
        spec.input("colvar_targets", valid_type=Str)
        spec.input("target_unit", valid_type=Str)
        spec.input("spring", valid_type=Float, default=Float(75.0))
        spec.input("spring_unit", valid_type=Str)

        spec.input("subsys_colvar", valid_type=ParameterData)
        spec.input("calc_type", valid_type=Str)

        spec.outline(
            cls.init,
            while_(cls.next_replica)(
                cls.generate_replica,
                while_(cls.not_converged)(
                    cls.generate_replica
                ),
                cls.store_replica
            )
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        try:
            self.ctx.remote_calc_folder = self.ctx.replica.out.remote_folder
            self.ctx.structure = self.ctx.replica.out.output_structure
            self.report('Convergence check: {}'.format(self.ctx.replica.res.exceeded_walltime))
            return self.ctx.replica.res.exceeded_walltime
        except AttributeError:
            return True

    # ==========================================================================
    def init(self):
        self.report('Init generate replicas')

        self.ctx.replica_list = str(self.inputs.colvar_targets).split()
        self.ctx.replicas_done = 0
        self.ctx.this_name = self.inputs.replica_name

        self.report('#{} replicas'.format(len(self.ctx.replica_list)))

    # ==========================================================================
    def next_replica(self):
        self.report('Go to replica - {}'.format(len(self.ctx.replica_list)))
        self.report('Remaining list: {} ({})'.format(self.ctx.replica_list,
                                                    len(self.ctx.replica_list)))
        if len(self.ctx.replica_list) > 0:
            self.ctx.this_replica = self.ctx.replica_list.pop(0)
        else:
            return False

        if self.ctx.replicas_done == 0:
            self.ctx.remote_calc_folder = None
            self.ctx.structure = self.inputs.structure
        else:
            self.ctx.remote_calc_folder = self.ctx.replica.out.remote_folder
            self.ctx.structure = self.ctx.replica.out.output_structure

        self.ctx.replicas_done += 1

        return True

    # ==========================================================================
    def generate_replica(self):
        self.report("Running CP2K geometry optimization - Target: "
                    .format(self.ctx.this_replica))

        inputs = self.build_calc_inputs(self.ctx.structure,
                                        self.inputs.cell,
                                        self.inputs.cp2k_code,
                                        self.ctx.this_replica,
                                        self.inputs.fixed_atoms,
                                        self.inputs.num_machines,
                                        self.ctx.remote_calc_folder,
                                        self.ctx.this_name,
                                        self.inputs.spring,
                                        self.inputs.spring_unit,
                                        self.inputs.target_unit,
                                        self.inputs.subsys_colvar,
                                        self.inputs.calc_type)

        self.report(" ")
        self.report("inputs: "+str(inputs))
        self.report(" ")
        future = submit(Cp2kCalculation.process(), **inputs)
        self.report("future: "+str(future))
        self.report(" ")
        return ToContext(replica=Calc(future))

    # ==========================================================================
    def store_replica(self):
        return self.out('replica_{}_{}'.format(self.ctx.this_replica,
                                               self.ctx.this_name),
                        self.ctx.replica.out.output_structure)

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, structure, cell, code, colvar_target,
                          fixed_atoms, num_machines, remote_calc_folder,
                          replica_name, spring, spring_unit, target_unit,
                          subsys_colvar, calc_type):

        inputs = {}
        inputs['_label'] = "replica_geo_opt"
        inputs['_description'] = "replica_{}_{}".format(replica_name,
                                                        colvar_target)

        inputs['code'] = code
        inputs['file'] = {}

        # make sure we're really dealing with a gold slab
        atoms = structure.get_ase()  # slow
        try:
            first_slab_atom = np.argwhere(atoms.numbers == 79)[0, 0] + 1
            is_H = atoms.numbers[first_slab_atom-1:] == 1
            is_Au = atoms.numbers[first_slab_atom-1:] == 79
            assert np.all(np.logical_or(is_H, is_Au))
            assert np.sum(is_Au) / np.sum(is_H) == 4
        except AssertionError:
            raise Exception("Structure is not a proper slab.")

        # structure
        molslab_f, mol_f = cls.mk_coord_files(atoms, first_slab_atom)
        inputs['file']['molslab_coords'] = molslab_f
        inputs['file']['mol_coords'] = mol_f
        
        # Au potential
        pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
        inputs['file']['au_pot'] = pot_f

        # parameters
        # if no cell is given use the one from the xyz file.
        if cell == '' or len(str(cell)) < 3:
            cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                       atoms.cell[1, 1],
                                       atoms.cell[2, 2])
        else:
            cell_abc = cell
            
        remote_computer = code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        inp = cls.get_cp2k_input(cell_abc,
                                 colvar_target,
                                 fixed_atoms,
                                 spring, spring_unit,
                                 target_unit,
                                 subsys_colvar,
                                 calc_type,
                                 machine_cores*num_machines,
                                 first_slab_atom,
                                 len(atoms))

        if remote_calc_folder is not None:
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.xyz']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": 86000,
        }

        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc,
                       colvar_target, fixed_atoms,
                       spring, spring_unit, target_unit, subsys_colvar,
                       calc_type, machine_cores, first_slab_atom,
                       last_slab_atom):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'GEO_OPT',
                'WALLTIME': 85500,
                'PRINT_LEVEL': 'LOW'
            },
            'MOTION': cls.get_motion(colvar_target, fixed_atoms, spring,
                                     spring_unit, target_unit),
            'FORCE_EVAL': [],
        }
        
        if calc_type == 'Mixed DFTB':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell_abc,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores,
                                                      subsys_colvar),
                                 cls.force_eval_fist(cell_abc),
                                 cls.get_force_eval_qs_dftb(cell_abc)]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Mixed DFT':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell_abc,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores,
                                                      subsys_colvar),
                                 cls.force_eval_fist(cell_abc),
                                 cls.get_force_eval_qs_dft(cell_abc, only_molecule=True)]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Full DFT':
            inp['FORCE_EVAL'] = [cls.get_force_eval_qs_dft(cell_abc, only_molecule=False,
                                                           subsys_colvar=subsys_colvar)]
        return inp

    # ==========================================================================
    @classmethod
    def force_eval_mixed(cls, cell_abc, first_slab_atom, last_slab_atom,
                         machine_cores, subsys_colvar):
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
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
                'COLVAR': subsys_colvar.get_attrs()
            }
        }

        return force_eval
    
    # ==========================================================================
    @classmethod
    def force_eval_fist(cls, cell_abc):
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
                    'ABC': cell_abc,
                },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            },
        }
        return force_eval
    
    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dftb(cls, cell_abc):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'QS': {
                    'METHOD': 'DFTB',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
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
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORDINATE': 'xyz'
                }
            }
        }

        return force_eval
    
    # ==========================================================================
    @classmethod
    def get_motion(cls, colvar_target, fixed_atoms, spring, spring_unit,
                   target_unit):
        motion = {
            'CONSTRAINT': {
                'COLLECTIVE': {
                    'COLVAR': 1,
                    'RESTRAINT': {
                        'K': '[{}] {}'.format(spring_unit, spring)
                    },
                    'TARGET': '[{}] {}'.format(target_unit, colvar_target),
                    'INTERMOLECULAR': ''
                },
                'FIXED_ATOMS': {
                    'LIST': '{}'.format(fixed_atoms)
                }
            },
            'GEO_OPT': {
                'MAX_FORCE': '0.0001',
                'MAX_ITER': '5000',
                'OPTIMIZER': 'LBFGS'
            },
        }

        return motion

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, only_molecule,
                              subsys_colvar=None):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'RESTART_FILE_NAME': './parent_calc/aiida-RESTART.wfn',
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '600',
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '20',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-7',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '15',
                        'EPS_SCF': '1.0E-7',
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
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'xyz',
                },
                'KIND': [],
            }
        }
        if only_molecule:
            force_eval['SUBSYS']['TOPOLOGY']['COORD_FILE_NAME'] = 'mol.xyz'

        if subsys_colvar is not None:
            force_eval['SUBSYS']['COLVAR'] = subsys_colvar.get_attrs()
        
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
    
    # ==========================================================================
    @classmethod
    def mk_coord_files(cls, atoms, first_slab_atom):
        mol = atoms[:first_slab_atom-1]

        tmpdir = tempfile.mkdtemp()
        molslab_fn = tmpdir + '/mol_on_slab.xyz'
        mol_fn = tmpdir + '/mol.xyz'

        atoms.write(molslab_fn)
        mol.write(mol_fn)

        molslab_f = SinglefileData(file=molslab_fn)
        mol_f = SinglefileData(file=mol_fn)

        shutil.rmtree(tmpdir)

        return molslab_f, mol_f
