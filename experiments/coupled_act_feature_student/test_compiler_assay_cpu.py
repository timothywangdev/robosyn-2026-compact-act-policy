import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before assay CPU tests')
import unittest
import torch
from .compiler_assay import fixed_indices,validate_training,CANDIDATES,correction_precision,numeric_output_difference


class Tests(unittest.TestCase):
    def test_exact_current_constructor_order_and_camera_frame_mismatch(self):
        tasks=[str(i) for i in range(10)]
        plan=dict(tasks=tasks,rows=[dict(task=task,partition='optimizer',clip_id=task+'current') for task in tasks])
        ctor=dict(plan='same',records=[dict(clip_id=t+'current') for t in tasks]);parity=dict(plan='same',metrics=[dict(task=t,clip_id=t+'current') for t in tasks])
        self.assertEqual(fixed_indices(plan,ctor,parity),list(range(10)))
        parity['metrics'][4]['clip_id']='different_current_frame'
        with self.assertRaises(ValueError):fixed_indices(plan,ctor,parity)
    def test_actual_fixed_four_receipt_rejects_partial_or_wrong_updates(self):
        r=dict(schema='actual_paired_ACT_feature_student_distill_and_policy_aware_GPU_v2',status='fixed30_four_candidates_complete_exploratory_no_robot_promotion',robot_quality_promoted=False,cache='same',results={k:dict(optimizer_updates=210) for k in CANDIDATES})
        validate_training(r,dict(cache='same'));r['results'][CANDIDATES[-1]]['optimizer_updates']=209
        with self.assertRaises(ValueError):validate_training(r,dict(cache='same'))
    def test_actual_inference_TF32_scope_preserves_cudnn_enabled_and_restores(self):
        before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.enabled)
        with correction_precision(torch):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32);self.assertFalse(torch.backends.cudnn.allow_tf32);self.assertEqual(torch.backends.cudnn.enabled,before[2])
        self.assertEqual(before,(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.enabled))
        self.assertFalse(torch.cuda.is_initialized())
    def test_difference_reports_gate_change_without_reclassifying_numeric_parity(self):
        a=dict(accepted=torch.tensor([[True,False]]),native_actions=torch.tensor([1.]))
        b=dict(accepted=torch.tensor([[False,False]]),native_actions=torch.tensor([1.0001]))
        row=numeric_output_difference(a,b);self.assertFalse(row['accepted']['bitexact']);self.assertGreater(row['native_actions']['maxabs'],0)


if __name__=='__main__':unittest.main()
