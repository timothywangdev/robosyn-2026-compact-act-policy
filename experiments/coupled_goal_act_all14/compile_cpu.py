"""Genuine CPU trace-path check; trained GPU acceleration remains unqualified."""
import argparse
import torch
from .model import CoupledGoalACT
from .test_cpu import fixture
from .compiler import compile_and_compare
from .common import write,require


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1701);m=CoupledGoalACT().eval().requires_grad_(False)
    with torch.no_grad():m.output.weight.normal_(0,.03);m.output.bias.fill_(.05);m.reliability[-1].bias.copy_(torch.tensor([4.,-4.]))
    _,result=compile_and_compare(m,fixture(batch_size=1),state_action_only=False,backend='aot_eager',warm_calls=3)
    require(not torch.cuda.is_initialized(),'CPU-only compiler-path check required')
    result.update(schema='coupled-goal-ACT-all14-actual-CPU-compiled-subpaths-v1',CUDA_initialized=False,
                  GPU_queries=0,GPU_launches=0,trained_model_qualification=False)
    write(a.output,result);print('actual_CPU_compiled_paths_called_and_parity_pass_no_GPU_acceleration_claim',flush=True)


if __name__=='__main__':main()
