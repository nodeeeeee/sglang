"""Distribute small-batch BF16 RMSNorm rows across independent CTAs."""
import functools
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel

@functools.cache
def compiled():
    obj=FusedAddRMSNormKernel(cutlass.BFloat16,2880,0.0,sm_version=100)
    obj.num_threads=32
    obj.rows_per_block=1
    m=cute.sym_int(64)
    x=cute.runtime.make_fake_compact_tensor(cutlass.BFloat16,(m,2880),stride_order=(1,0),assumed_align=128)
    r=cute.runtime.make_fake_compact_tensor(cutlass.BFloat16,(m,2880),stride_order=(1,0),assumed_align=128)
    w=cute.runtime.make_fake_compact_tensor(cutlass.BFloat16,(2880,),assumed_align=16)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(obj,x,r,w,Int64(1),Float32(1e-5),True,stream,options='--enable-tvm-ffi')

def direct_norm(x,res,w,eps):
    if x.shape[0] <= 128:
        compiled()(x,res,w,x.shape[0],eps)
    else:
        from flashinfer.norm import fused_add_rmsnorm
        fused_add_rmsnorm(x,res,w,eps)


def try_fused_add_rmsnorm(x, residual, weight, eps):
    if not (
        x.is_cuda and x.ndim == 2 and x.shape[1] == 2880
        and 1 < x.shape[0] <= 128 and x.dtype == torch.bfloat16
        and residual.shape == x.shape and residual.dtype == x.dtype
        and weight.shape == (2880,) and weight.dtype == x.dtype
        and x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
        and x.data_ptr() % 128 == 0 and residual.data_ptr() % 128 == 0
        and weight.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(x.device) == (10, 0)
    ):
        return False
    direct_norm(x, residual, weight, eps)
    return True
