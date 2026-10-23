"""Negative reference tests and exact frozen-runtime transport delta check."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from .prepare import ROOT,ref
from .resolver import Resolver
from .contracts import InventoryPath


class Transport(unittest.TestCase):
    def fixture(self,root):
        path=root/'installed.json';path.write_text('{"raw":true}\n')
        row=ref(path);row['path']='/historical/not-installed/proof.json'
        return path,row,[dict(original=row,installed='installed.json')]

    def test_original_bytes_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root);before=path.read_bytes();r=Resolver(root,b)
            self.assertEqual(r.verify_ref(row),path);self.assertEqual(path.read_bytes(),before)
            self.assertEqual(row['path'],'/historical/not-installed/proof.json')

    def test_sha_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root);r=Resolver(root,b);r.verify_ref(row)
            path.write_text('{"raw":false}')
            with self.assertRaises(ValueError):r.verify_ref(row)

    def test_unbound_or_wrong_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root);r=Resolver(root,b)
            for key,value in [('path','/different'),('bytes',row['bytes']+1),('sha256','0'*64)]:
                with self.assertRaises(ValueError):r.verify_ref(dict(row,**{key:value}))

    def test_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root)
            for installed in ('../outside','/absolute'):
                with self.assertRaises(ValueError):Resolver(root,[dict(original=row,installed=installed)])
            link=root/'link';link.symlink_to(path)
            r=Resolver(root,[dict(original=row,installed='link')])
            with self.assertRaises(ValueError):r.verify_ref(row)

    def test_conflicting_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root)
            with self.assertRaises(ValueError):Resolver(root,b+[dict(original=row,installed='different')])

    def test_two_original_paths_one_exact_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,row,b=self.fixture(root);other=dict(row,path='/second/historical/path')
            r=Resolver(root,b+[dict(original=other,installed='installed.json')])
            self.assertEqual(r.verify_ref(row),r.verify_ref(other))

    def test_model_runtime_only_observer_module_path_changed(self):
        a=(ROOT/'experiments/coupled_goal_act_all14_runtime/runtime.py').read_text()
        b=(ROOT/'experiments/coupled_goal_act_all14_submission/runtime_native.py').read_text()
        self.assertEqual(a.replace('experiments.coupled_goal_act_all14_runtime.observer_worker','experiments.coupled_goal_act_all14_submission.observer_worker'),b)
        a=(ROOT/'experiments/coupled_goal_act_all14_runtime/observer_worker.py').read_text()
        b=(ROOT/'experiments/coupled_goal_act_all14_submission/observer_worker.py').read_text();self.assertEqual(a,b)


if __name__=='__main__':unittest.main()
