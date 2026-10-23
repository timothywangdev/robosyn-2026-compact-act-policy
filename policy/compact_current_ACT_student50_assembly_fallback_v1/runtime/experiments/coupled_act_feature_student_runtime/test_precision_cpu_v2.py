"""Actual flag scopes and pixel/command buffer hashing, without CUDA/model loads."""
from types import SimpleNamespace
import unittest
import numpy as np
from .runtime_v2 import native_ACT_precision,buffer_proof
from experiments.coupled_goal_act_all14_runtime.runtime import Runtime as TeacherRuntime

class Tests(unittest.TestCase):
    def fake(self):
        return SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),cudnn=SimpleNamespace(allow_tf32=False)))
    def test_nested_correction_restores_native_flags(self):
        torch=self.fake();fake=SimpleNamespace(torch=torch)
        with native_ACT_precision(torch):
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),(False,True))
            with TeacherRuntime.correction_precision(fake):
                self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),(False,False))
            self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),(False,True))
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),(True,False))
    def test_exceptions_restore_flags(self):
        torch=self.fake();fake=SimpleNamespace(torch=torch)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with native_ACT_precision(torch):
                with TeacherRuntime.correction_precision(fake):raise RuntimeError('fixture')
        self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),(True,False))
    def test_pixel_buffer_proof_dtype_shape_and_changes(self):
        image=np.zeros((480,640,3),np.uint8);before=image.copy();a=buffer_proof(image)
        self.assertEqual(a,buffer_proof(image.copy()));self.assertEqual(a['dtype'],'uint8');self.assertEqual(a['shape'],[480,640,3])
        np.testing.assert_array_equal(before,image);image[0,0,0]=1
        self.assertNotEqual(a['sha256'],buffer_proof(image)['sha256'])
    def test_exact_float32_actions_and_qpos(self):
        values=np.arange(14,dtype=np.float32);before=values.copy();proof=buffer_proof(values)
        self.assertEqual(proof,buffer_proof(values.copy()));self.assertEqual(proof['shape'],[14]);np.testing.assert_array_equal(values,before)
        self.assertNotEqual(proof['sha256'],buffer_proof(values.astype(np.float64))['sha256'])

if __name__=='__main__':unittest.main()
