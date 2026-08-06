import importlib.util
import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "resource_plan", TOOLS / "resource_plan.py"
)
resource_plan = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = resource_plan
SPEC.loader.exec_module(resource_plan)


class ResourcePlanTests(unittest.TestCase):
    def config(self):
        return {
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "full_attention_interval": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "vocab_size": 248320,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 64,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        }

    def test_397b_state_geometry(self):
        plan = resource_plan.plan_resources(
            self.config(),
            slots=4,
            context=4096,
            system_ram_gib=64,
            gpu_gib=64,
            disk_free_gib=500,
            include_mtp=False,
        )
        state = plan["state_bytes"]
        self.assertEqual(plan["architecture"]["linear_layers"], 45)
        self.assertEqual(plan["architecture"]["full_layers"], 15)
        self.assertEqual(
            state["recurrent_per_slot"], 45 * 64 * 128 * 128 * 4
        )
        self.assertEqual(
            state["conv_per_slot"], 45 * (2 * 16 * 128 + 64 * 128) * 4 * 4
        )
        self.assertEqual(state["kv_per_token_per_slot"], 15 * 2 * 2 * 256 * 4)
        expected_snapshot = (
            state["recurrent_per_slot"]
            + state["conv_per_slot"]
            + 4096 * 4
            + 4096 * state["kv_per_token_per_slot"]
        )
        self.assertEqual(state["snapshot_per_slot_at_context"], expected_snapshot)
        self.assertEqual(
            state["sequential_switch_bytes_per_token_per_active_slot"],
            2 * expected_snapshot,
        )
        self.assertGreater(plan["container_bytes"]["routed_experts"], 190 * 1024**3)
        self.assertLess(plan["container_bytes"]["routed_experts"], 193 * 1024**3)

    def test_guards_fail_honestly(self):
        plan = resource_plan.plan_resources(
            self.config(),
            system_ram_gib=8,
            gpu_gib=8,
            disk_free_gib=100,
        )
        self.assertFalse(plan["safe"])
        self.assertEqual(
            plan["checks"], {"disk": False, "ram": False, "vram": False}
        )

    def test_ram_plan_retains_dense_weights_and_reports_cache_geometry(self):
        plan = resource_plan.plan_resources(
            self.config(),
            slots=1,
            context=4096,
            cuda_expert_gib=6,
            ram_cache_gib=18,
            system_ram_gib=32,
            gpu_gib=16,
            disk_free_gib=500,
            runtime_headroom_gib=1,
            include_mtp=False,
        )
        tier = plan["tier_plan"]
        state = plan["state_bytes"]["total"]
        self.assertEqual(
            tier["ram_required"],
            tier["dense_shared_bytes"] + 18 * 1024**3 + state + 1024**3,
        )
        self.assertEqual(tier["ram_expert_cache_slots_per_layer"], 48)
        self.assertGreater(tier["cuda_expert_cache_slots_per_layer_average"], 16)
        self.assertLess(tier["nominal_combined_expert_coverage"], 0.13)
        self.assertGreater(tier["nominal_combined_expert_coverage"], 0.12)

    def test_mtp_increases_container_and_kv(self):
        base = resource_plan.plan_resources(
            self.config(),
            system_ram_gib=128,
            gpu_gib=128,
            disk_free_gib=500,
            include_mtp=False,
        )
        mtp = resource_plan.plan_resources(
            self.config(),
            system_ram_gib=128,
            gpu_gib=128,
            disk_free_gib=500,
            include_mtp=True,
        )
        self.assertGreater(mtp["container_bytes"]["total"], base["container_bytes"]["total"])
        self.assertEqual(
            mtp["state_bytes"]["kv_per_token_per_slot"]
            - base["state_bytes"]["kv_per_token_per_slot"],
            2 * 2 * 256 * 4,
        )


if __name__ == "__main__":
    unittest.main()
