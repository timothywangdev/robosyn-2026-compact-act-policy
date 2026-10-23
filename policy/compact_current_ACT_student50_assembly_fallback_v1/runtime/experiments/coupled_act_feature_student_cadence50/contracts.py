"""All original trained-evidence predicates with an explicit executor50 schema."""
from experiments.coupled_act_feature_student_runtime import contracts as original
from .derive import derive
require,ref,verify,verify_ref,module=original.require,original.ref,original.verify,original.verify_ref,original.module
STATS,HORIZONS,CAMERAS=original.STATS,original.HORIZONS,original.CAMERAS
validate_saved=original.validate_saved
SCHEMA='coupled-ACT-current-map-student-native50-execute50-v1'
load_manifest=derive(original.load_manifest,'contract',{'SCHEMA':SCHEMA})
