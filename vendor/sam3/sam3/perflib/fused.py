# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    # Follow the input dtype instead of hardcoding bfloat16. Under bf16 autocast
    # this is a no-op (upstream behavior); on XPU/CPU in fp32 it keeps the fused
    # kernel dtype-consistent with the rest of the model (fp32).
    compute_dtype = mat1.dtype
    if compute_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        compute_dtype = torch.float32
    self = linear.bias.detach().to(compute_dtype)
    mat2 = linear.weight.detach().to(compute_dtype)
    mat1 = mat1.to(compute_dtype)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
    raise ValueError(f"Unexpected activation {activation}")
