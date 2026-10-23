"""Additive native CNN precision scope and source-buffer pairing evidence."""
from contextlib import contextmanager
import hashlib
import json
from .contracts import require,verify_ref
from .runtime import Runtime as OriginalRuntime,eval,reset_model,close_model


@contextmanager
def native_ACT_precision(torch):
    matmul=torch.backends.cuda.matmul.allow_tf32
    cudnn=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=matmul
        torch.backends.cudnn.allow_tf32=cudnn


def buffer_proof(value):
    import numpy as np
    value=np.ascontiguousarray(value)
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())


class PrecisionObservedRuntime:
    def select_action(self,batch):
        require(self.cfg['native_ACT_precision']==dict(matmul_allow_tf32=False,cudnn_allow_tf32=True)
                and self.cfg['pairing_initial_current_buffers_logged'] is True,
                'Explicit native CNN precision and current-source pairing evidence required')
        if self.executed==0:
            from experiments.goal_latent_act_sequence_runtime_full.current_snapshot import current_inputs
            observer,_=current_inputs(batch,json.loads(verify_ref(self.cfg['assets']['state_evidence']).read_text()))
            q=observer['qpos']
            proof=dict(task=self.cfg['task'],mode=self.cfg['mode'],episode=self.episode,
                current_qpos14=q.tolist(),qpos_buffer=buffer_proof(q),
                RGB={camera:buffer_proof(observer[camera]) for camera in ('cam_high','cam_right_wrist','cam_left_wrist')},
                input_features_changed=False,hashes_are_features=False)
            print('[student initial current buffers] '+json.dumps(proof),flush=True)
        with native_ACT_precision(self.torch):
            # The frozen delegate scopes only the correction to false/false;
            # every native ACT forward therefore sees false/true.
            action=super().select_action(batch)
        if self.executed<=7:
            print('[student early action buffer] '+json.dumps(dict(task=self.cfg['task'],mode=self.cfg['mode'],
                episode=self.episode,issued=self.executed,action=buffer_proof(action.detach().cpu().numpy()))),flush=True)
        return action


class Runtime(PrecisionObservedRuntime,OriginalRuntime):pass
def get_model(args):return Runtime(args)
