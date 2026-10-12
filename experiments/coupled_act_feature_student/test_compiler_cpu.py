import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Hide CUDA before CPU compiler tests')
import unittest
import torch
from .model import Student
from .compiler import PureStudent,validate,validated,qualify


class Tests(unittest.TestCase):
    def test_real_four_mode_AOT_paths_and_validation_are_preserved(self):
        torch.set_num_threads(2);torch.manual_seed(1701)
        inputs=dict(visual=torch.randn(1,3,512,2,3),qpos_features=torch.randn(1,14),visual_global=torch.randn(1,3,512))
        for pooling in ('spatial2x3','global'):
            for qpos_only in (False,True):
                model=Student(pooling,qpos_only).eval();pure=PureStudent(model)
                with torch.inference_mode():
                    expected=model(**inputs);actual=validated(model,pure,inputs)
                for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]))
                def fixture_tail(value,index):
                    return dict(native_actions=torch.cat((value['projected_roles'][:,:7],value['predicted_goal_delta'][:,:7]),-1),accepted=value['projected_roles'][:,:2]>=.5)
                call,report=qualify(model,[inputs],backend='aot_eager',benchmark=False,postprocess=fixture_tail)
                self.assertFalse(report['acceleration_qualified']);self.assertIsNone(report['median_speedup'])
                self.assertEqual(len(report['numerical']),1)
                self.assertTrue(report['actual_tail_numeric_and_Boolean_parity_required'])
                bad=dict(inputs,visual=inputs['visual'].clone());bad['visual'][0,0,0,0,0]=float('nan')
                with self.assertRaises(ValueError):validated(model,call,bad)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
