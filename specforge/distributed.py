from datetime import timedelta

import einops
import torch
import torch.distributed as dist

from specforge.utils import print_on_rank0, print_with_rank

_DEVICE_MESH = None
_TARGET_TP_DEVICE_MESH = None
_DRAFT_DP_DEVICE_MESH = None
_TARGET_TP_GROUP = None
_TARGET_DP_GROUP = None
_DRAFT_TP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_CP_GROUP = None


def get_target_tp_group():
    global _TARGET_TP_GROUP
    return _TARGET_TP_GROUP


def get_target_tp_size():
    global _TARGET_TP_GROUP
    if _TARGET_TP_GROUP is None:
        return 1
    return dist.get_world_size(_TARGET_TP_GROUP)


def get_target_tp_rank():
    global _TARGET_TP_GROUP
    if _TARGET_TP_GROUP is None:
        return 0
    return dist.get_rank(_TARGET_TP_GROUP)


def get_target_dp_group():
    global _TARGET_DP_GROUP
    return _TARGET_DP_GROUP


def get_target_dp_size():
    global _TARGET_DP_GROUP
    if _TARGET_DP_GROUP is None:
        return 1
    return dist.get_world_size(_TARGET_DP_GROUP)


def get_target_dp_rank():
    global _TARGET_DP_GROUP
    if _TARGET_DP_GROUP is None:
        return 0
    return dist.get_rank(_TARGET_DP_GROUP)


def get_draft_tp_group():
    global _DRAFT_TP_GROUP
    return _DRAFT_TP_GROUP


def get_draft_tp_size():
    global _DRAFT_TP_GROUP
    if _DRAFT_TP_GROUP is None:
        return 1
    return dist.get_world_size(_DRAFT_TP_GROUP)


def get_draft_tp_rank():
    global _DRAFT_TP_GROUP
    if _DRAFT_TP_GROUP is None:
        return 0
    return dist.get_rank(_DRAFT_TP_GROUP)


def get_draft_cp_group():
    global _DRAFT_CP_GROUP
    return _DRAFT_CP_GROUP


def get_draft_cp_size():
    global _DRAFT_CP_GROUP
    if _DRAFT_CP_GROUP is None:
        return 1
    return dist.get_world_size(_DRAFT_CP_GROUP)


def get_draft_cp_rank():
    global _DRAFT_CP_GROUP
    if _DRAFT_CP_GROUP is None:
        return 0
    return dist.get_rank(_DRAFT_CP_GROUP)


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_dp_size():
    global _DRAFT_DP_GROUP
    if _DRAFT_DP_GROUP is None:
        return 1
    return dist.get_world_size(_DRAFT_DP_GROUP)


def get_draft_dp_rank():
    global _DRAFT_DP_GROUP
    if _DRAFT_DP_GROUP is None:
        return 0
    return dist.get_rank(_DRAFT_DP_GROUP)


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_target_tp_device_mesh():
    global _TARGET_TP_DEVICE_MESH
    return _TARGET_TP_DEVICE_MESH


def init_distributed(
    timeout: int = 10,
    target_tp_size: int = 1,
    draft_tp_size: int = 1,
    draft_cp_size: int = 1,
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    target_dp_size = world_size // target_tp_size
    draft_dp_size = world_size // (draft_tp_size * draft_cp_size)
    assert (
        world_size == target_tp_size * target_dp_size
    ), "world size must be divisible by target tp size"
    assert (
        world_size == draft_tp_size * draft_cp_size * draft_dp_size
    ), "world size must be divisible by draft tp size and draft cp size"
    target_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (target_dp_size, target_tp_size),
        mesh_dim_names=["target_dp", "target_tp"],
    )
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, draft_tp_size, draft_cp_size),
        mesh_dim_names=["draft_dp", "draft_tp", "draft_cp"],
    )
    print_on_rank0(f"target device mesh: {target_device_mesh}")
    print_on_rank0(f"draft device mesh: {draft_device_mesh}")
    global _TARGET_TP_GROUP, _TARGET_DP_GROUP, _DRAFT_TP_GROUP, _DRAFT_DP_GROUP, _DRAFT_CP_GROUP, _TARGET_TP_DEVICE_MESH
    _TARGET_TP_GROUP = target_device_mesh.get_group("target_tp")
    _TARGET_DP_GROUP = target_device_mesh.get_group("target_dp")
    _DRAFT_TP_GROUP = draft_device_mesh.get_group("draft_tp")
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_CP_GROUP = draft_device_mesh.get_group("draft_cp")
    _TARGET_TP_DEVICE_MESH = dist.DeviceMesh.from_group(
        _TARGET_TP_GROUP, device_type="cuda"
    )


def destroy_distributed():
    global _TARGET_TP_GROUP, _TARGET_DP_GROUP, _DRAFT_TP_GROUP, _DRAFT_DP_GROUP
    dist.destroy_process_group(_TARGET_TP_GROUP)
    dist.destroy_process_group(_TARGET_DP_GROUP)
    dist.destroy_process_group(_DRAFT_TP_GROUP)
    dist.destroy_process_group(_DRAFT_DP_GROUP)
    dist.destroy_process_group()


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gather_tensor = torch.cat(obj_list, dim=dim)
    return gather_tensor


@torch.compiler.disable()
def _all_to_all_single(output_tensor, input_tensor, group):
    # Disable compilation since torch compile changes contiguity.
    assert input_tensor.is_contiguous(), "Input tensor must be contiguous."
    assert output_tensor.is_contiguous(), "Output tensor must be contiguous."
    return dist.all_to_all_single(output_tensor, input_tensor, group=group)


class CollectTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup, num_heads: int):
        """Redistribute heads and receive tokens.
        Args:
            tensor: query, key or value. Shape: [B, S // CP, num_heads * head_dim]
        Returns:
            tensor: shape: [B, S, local_heads, head_dim]
        local_heads = num_heads // CP
        """
        ctx.group = group
        ctx.num_heads = num_heads
        cp_size = dist.get_world_size(group)
        assert num_heads % cp_size == 0
        ctx.local_heads = num_heads // cp_size
        ctx.cp_size = cp_size

        tensor = einops.rearrange(
            tensor,
            "B S (CP h d) -> CP S h B d",
            CP=cp_size,
            h=ctx.local_heads,
        ).contiguous()

        output_chunks = torch.empty_like(tensor)
        _all_to_all_single(output_chunks, tensor, group=group)
        return einops.rearrange(output_chunks, "CP S h B d -> B (CP S) h d")

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward: Invert the operations in forward.
        """
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        # Reverse the second rearrange:
        rearranged_grad = einops.rearrange(
            grad_out, "B (CP S) h d -> CP S h B d", CP=ctx.cp_size, h=ctx.local_heads
        ).contiguous()

        comm_grad = torch.empty_like(rearranged_grad).contiguous()
        _all_to_all_single(comm_grad, rearranged_grad, ctx.group)

        grad_input = einops.rearrange(
            comm_grad, "CP S h B d -> B S (CP h d)"
        ).contiguous()
        return grad_input, None, None


def ulysses_collect_tokens(
    tensor: torch.Tensor, num_heads: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    if not cp_group or dist.get_world_size(cp_group) == 1:
        return tensor
    return CollectTokens.apply(tensor, cp_group, num_heads)


class CollectHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup):
        """
        Redistribute tokens and receive heads.

        Args:
            x: Output of attention. Shape: [B, N, local_heads, head_dim]
        Returns:
            Shape: [B, M, num_heads * head_dim]
        """
        ctx.group = group
        ctx.local_heads = tensor.size(2)
        ctx.head_dim = tensor.size(3)
        group_size = dist.get_world_size(group)
        ctx.group_size = group_size  # save for backward
        tensor = einops.rearrange(
            tensor, "B (CP S) h d -> CP S h B d", CP=group_size
        ).contiguous()
        output = torch.empty_like(tensor)
        _all_to_all_single(output, tensor, group)
        del tensor
        return einops.rearrange(output, "CP S h B d -> B S (CP h d)")

    @staticmethod
    def backward(ctx, grad_out):
        rearranged_grad = einops.rearrange(
            grad_out,
            "B S (CP h d) -> CP S h B d",
            CP=ctx.group_size,
            h=ctx.local_heads,
            d=ctx.head_dim,
        ).contiguous()
        comm_grad = torch.empty_like(rearranged_grad).contiguous()
        _all_to_all_single(comm_grad, rearranged_grad, ctx.group)
        grad_input = einops.rearrange(comm_grad, "CP S h B d -> B (CP S) h d")
        return grad_input, None


def ulysses_collect_heads(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    if not cp_group or dist.get_world_size(cp_group) == 1:
        return x.contiguous()
    return CollectHeads.apply(x, cp_group).contiguous()
