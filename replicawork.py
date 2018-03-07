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


class ReplicaWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(ReplicaWorkchain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("num_machines", valid_type=Int, default=Int(54))
        spec.input("cell", valid_type=Str, default=Str(''))
        spec.input("colvar_targets", valid_type=Str)
        spec.input("colvar_type", valid_type=Str)
        spec.input("colvar_atoms", valid_type=Str)
        spec.input("replica_name", valid_type=Str)
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
        spec.input("spring", valid_type=Float, default=Float(75.0))

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
                                        self.inputs.colvar_type,
                                        self.inputs.colvar_atoms,
                                        self.inputs.fixed_atoms,
                                        self.inputs.num_machines,
                                        self.ctx.remote_calc_folder,
                                        self.ctx.this_name,
                                        self.inputs.spring)

        #self.report(" ")
        #self.report("inputs: "+str(inputs))
        #self.report(" ")
        future = submit(Cp2kCalculation.process(), **inputs)
        self.report("future: "+str(future))
        return ToContext(replica=Calc(future))

    # ==========================================================================
    def store_replica(self):
        return self.out('replica_{}_{}'.format(self.ctx.this_replica,
                                               self.ctx.this_name),
                        self.ctx.replica.out.output_structure)

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, structure, cell, code, colvar_target,
                          colvar_type, colvar_atoms, fixed_atoms,
                          num_machines, remote_calc_folder, replica_name,
                          spring):

        inputs = {}
        inputs['_label'] = "replica_geo_opt"
        inputs['_description'] = "replica_{}_{}".format(replica_name,
                                                        colvar_target)

        inputs['code'] = code
        inputs['file'] = {}

        # write the xyz structure file
        tmpdir = tempfile.mkdtemp()

        atoms = structure.get_ase()  # slow
        molslab_fn = tmpdir + '/mol_on_slab.xyz'
        atoms.write(molslab_fn)
        molslab_f = SinglefileData(file=molslab_fn)
        inputs['file']['molslab_coords'] = molslab_f

        shutil.rmtree(tmpdir)

        # parameters
        # cell_abc = "41.637276  41.210215  40.000000"
        if cell == '' or len(str(cell)) < 3:
            cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                       atoms.cell[1, 1],
                                       atoms.cell[2, 2])
        else:
            cell_abc = cell

        inp = cls.get_cp2k_input(cell_abc,
                                 colvar_target,
                                 colvar_type,
                                 colvar_atoms,
                                 fixed_atoms,
                                 spring)

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
                       colvar_target, colvar_type, colvar_atoms, fixed_atoms,
                       spring):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'GEO_OPT',
                'WALLTIME': 85500,
                'PRINT_LEVEL': 'LOW'
            },
            'MOTION': cls.get_motion(colvar_target, fixed_atoms, spring),
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc,
                                                    colvar_type,
                                                    colvar_atoms),
        }
        return inp

    # ==========================================================================
    @classmethod
    def get_motion(cls, colvar_target, fixed_atoms, spring):
        motion = {
            'CONSTRAINT': {
                'COLLECTIVE': {
                    'COLVAR': 1,
                    'RESTRAINT': {
                        'K': '[eV/angstrom^2] {}'.format(spring)
                    },
                    'TARGET': '[angstrom] {}'.format(colvar_target),
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
    def get_force_eval_qs_dft(cls, cell_abc, colvar_type, colvar_atoms):
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
                'COLVAR': {
                    str(colvar_type): {
                        'ATOMS': colvar_atoms
                    }
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

