"""Explicit eager correction after measured compiler attempt; same actor math."""
import json
from .contracts import load_manifest,require,verify_ref
from .runtime import Runtime as CompiledRuntime,eval,reset_model,close_model


EAGER='qualified_eager_after_measured_compile_attempt'


def load_eager_manifest(path,expected):
    cfg=load_manifest(path,expected)
    require(cfg['correction_runtime_mode']==EAGER and cfg['correction_partial_compile'] is False
        and cfg['require_compile_speedup'] is False,'Explicit separately prepared eager runtime required')
    required={'experiments/coupled_goal_act_all14_runtime/'+name+'.py' for name in
        ('runtime_eager','native_eager','controller_eager','prepare_eager','policy_shim_eager')}
    require(required<=set(cfg['source_sha256']),'Eager runtime source closure missing')
    if cfg['mode']!='parent':
        assets=cfg['assets'];attempt=json.loads(verify_ref(assets['compiler_attempt']).read_text())
        require(attempt['schema']=='coupled_all14_actual_all10_FIT_current_GPU_compiler_assay_v1'
            and attempt['status']=='both_trained_modes_all10_FIT_current_partial_Inductor_numeric_pass_no_robot_promotion'
            and set(attempt['branches'])=={'role_goal','state_action_control'},
            'Complete actual both-mode compiler evidence required; failed attempt cannot qualify this eager packet')
        for result in attempt['branches'].values():
            require(len(result['all10_same_input_rows'])==10 and result['compiler']['backend']=='inductor'
                and result['compiler']['acceleration_qualified'] is False
                and all(row['graphs']>0 and row['calls']>=6 for row in result['compiler']['graph_and_kernel_calls'].values()),
                'Eager choice requires both complete numerical gates and measured compiler slowdown')
        admitted=json.loads(verify_ref(attempt['admission']).read_text())
        branch='role_goal' if cfg['mode']=='goal' else 'state_action_control'
        require(admitted['weights'][branch]==assets['coupled_weights']
            and admitted['checkpoints'][branch]==assets['coupled_contract'],'Compiler attempted different actor bytes')
        wait=json.loads(verify_ref(assets['compiler_attempt_original_wait']).read_text())
        require(wait['schema']=='root_coupled_all14_compiler_actual_original_wait_v2'
            and wait['child_reaped'] is True and wait['signals_sent']==[] and wait['assay_receipt']==assets['compiler_attempt']
            and wait['original_child_exit_code']==0,
            'Literal attempted compiler original wait differs')
        drain=json.loads(verify_ref(assets['compiler_attempt_terminal_drain']).read_text())
        require(drain['schema']=='root_coupled_all14_compiler_original0_independent_drain_v2'
            and drain['original_wait']==assets['compiler_attempt_original_wait'] and drain['live_exact_birth_groups_sessions']==[]
            and drain['shared_lock_free'] is True and drain['signals_sent']==[]
            and len(drain['two_fresh_GPU_empty_snapshots'])==2
            and all(row['compute_stdout']=='' for row in drain['two_fresh_GPU_empty_snapshots']),
            'Attempted compiler must be naturally drained before eager runtime')
    return cfg


class Runtime(CompiledRuntime):
    def __init__(self,args):
        load_eager_manifest(args['coupled_all14_manifest'],args['coupled_all14_manifest_sha256'])
        super().__init__(args)

    def qualify_compiler(self,native,current):
        # The borrowed select_action still calls exactly the frozen scientific
        # inference function. This hook records a declared execution choice;
        # it neither replaces the actor nor invokes a compiler or alternate math.
        self.compiler_report=dict(execution_mode=EAGER,compiled=False,full_GRU_compiled=False,
            acceleration_qualified=False,compiler_attempt=self.cfg['assets']['compiler_attempt'],
            selection='root-declared eager candidate after actual attempted compilation; no silent fallback',
            current_native_world_compile_equivalence_claimed=False)
        print('[coupled all14 execution] '+json.dumps(self.compiler_report),flush=True)


def get_model(args):return Runtime(args)
