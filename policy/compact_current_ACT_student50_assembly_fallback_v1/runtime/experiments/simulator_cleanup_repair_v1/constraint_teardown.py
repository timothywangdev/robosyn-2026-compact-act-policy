"""Remove live fixed constraints through the existing SDK API before arena teardown."""
import ast
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

SDK_SHA256 = '5cde8b1a69fd6a6e4dd8623609925329239a0125ade9588a0e8d9767b8a8078a'
METHOD_AST_SHA256 = '1f96801db00965ac123b08f92957b4a18995b023f3fe2a465549cfeda51d81c5'


def _ast_fingerprint(node):
    def normalize(value):
        if isinstance(value, ast.AST):
            return dict(kind=type(value).__name__, fields={name:normalize(item)
                for name,item in ast.iter_fields(value) if name != 'type_params' or item})
        if isinstance(value, list): return [normalize(item) for item in value]
        return value
    return hashlib.sha256(json.dumps(normalize(node), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _constraint_first(original):
    """Change only teardown ordering; exceptions and original return propagate."""
    @wraps(original)
    def teardown(self):
        for name in list(self._constraints):
            if not self.remove_rigid_constraint(name):
                raise RuntimeError('Registered fixed constraint was not removed: '+str(name))
        if self._constraints:
            raise RuntimeError('Fixed constraint handles remain before arena teardown')
        return original(self)
    return teardown


def install(expected_sdk_file):
    expected = Path(expected_sdk_file).resolve(strict=True)
    if hashlib.sha256(expected.read_bytes()).hexdigest() != SDK_SHA256:
        raise ValueError('Pinned original SDK bytes differ; no cleanup patch installed')
    from embodichain.lab.sim import sim_manager
    if Path(sim_manager.__file__).resolve() != expected:
        raise ValueError('Imported SDK path differs from the original evaluator source root')
    manager = sim_manager.SimulationManager
    original = manager._deferred_destroy
    if Path(original.__code__.co_filename).resolve() != expected:
        raise ValueError('Original deferred cleanup is already replaced or comes from another source')
    method = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    digest = _ast_fingerprint(method)
    if digest != METHOD_AST_SHA256:
        raise ValueError('Original deferred cleanup AST differs')
    manager._deferred_destroy = _constraint_first(original)
    return dict(schema='prospective_original_SDK_constraint_first_cleanup_v1', sdk_sha256=SDK_SHA256,
                original_method_ast_sha256=METHOD_AST_SHA256,
                delta='Official remove_rigid_constraint before original deferred destruction',
                original_cleanup_still_called=True, actor_action_metric_or_episode_changed=False,
                native_qualification_required=True)
