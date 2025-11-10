# Filename: test_tp_correctness.py (Final version with tests for both MLP and Attention)

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate.utils import set_seed
from transformers import LlamaConfig

from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.draft.llama3_eagle import LlamaAttention

# === Core Parallel Test Functions ===


def run_attention_cp_test(rank, world_size, temp_dir_name):
    """This function executes the parallel computation for Attention and compares it with the 'golden standard'."""
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29504"
    init_distributed(draft_tp_size=1, draft_cp_size=world_size)
    torch.cuda.set_device(rank)
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=512,
    )
    attn_cp = LlamaAttention(config).cuda(rank)
    for name, param in attn_cp.named_parameters():
        dist.all_reduce(param, op=dist.ReduceOp.AVG)
    input_tensor = torch.randn(1, 16, config.hidden_size * 2).cuda(rank)
    dist.all_reduce(input_tensor, op=dist.ReduceOp.AVG)
    pos_ids = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(1, -1).cuda(rank)

    input_tensor_cp = input_tensor.chunk(world_size, dim=1)[rank]
    output_w_cp = attn_cp(input_tensor_cp, position_ids=pos_ids)

    attn_cp.cp_group = None
    attn_cp.cp_size = 1
    output_wo_cp = attn_cp(input_tensor, position_ids=pos_ids)
    output_wo_cp = output_wo_cp.chunk(world_size, dim=1)[rank]

    torch.testing.assert_close(output_w_cp, output_wo_cp, rtol=1e-4, atol=1e-5)
    dist.barrier()
    if rank == 0:
        print("✅ LlamaAttention CP correctness test passed!")
    destroy_distributed()


# === unittest Launcher ===
class TestTPCorrectness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_attention_correctness(self):
        world_size = 2
        temp_dir_path = self.temp_dir.name
        print("\n--- Running Attention TP Correctness Test ---")

        # Phase 1: Generate the "golden standard" for Attention
        set_seed(42)

        # Phase 2 & 3: Spawn parallel processes
        mp.spawn(
            run_attention_cp_test, nprocs=world_size, args=(world_size, temp_dir_path)
        )


if __name__ == "__main__":
    unittest.main()
