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

class Glm53ResourcePlanTests(unittest.TestCase):
    """GLM-5.3-Flash is planned with its own arithmetic.

    The Qwen planner raised KeyError('num_experts') on a GLM config, and the
    doctor is a hard preflight gate in LocalForge, so the lane could not start
    at all. Renaming fields would not have been enough either: GLM stores KV as
    an MLA latent (kv_lora_rank per token per full layer) where Qwen stores
    2 * kv_heads * head_dim, so Qwen's formula overstates KV by more than an
    order of magnitude.
    """

    def config(self):
        return {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "shiftwing_model_family": "glm-5.3",
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 45,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 2048,
                "intermediate_size": 12288,
                "vocab_size": 154880,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "v_head_dim": 256,
                "first_k_dense_replace": 3,
                "layer_types": ["linear_attention"] * 34 + ["deepseek_sparse_attention"] * 11,
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                },
            },
        }

    def plan(self, **overrides):
        arguments = dict(
            slots=1, context=16384, cuda_expert_gib=5.0, ram_cache_gib=12.0,
            system_ram_gib=30.0, gpu_gib=16.0, disk_free_gib=400.0,
            snapshot_bytes=190 * resource_plan.GIB,
        )
        arguments.update(overrides)
        return resource_plan.plan_resources(self.config(), **arguments)

    def test_glm_config_is_recognised(self):
        self.assertTrue(resource_plan._is_glm53(self.config()))
        self.assertEqual(self.plan()["family"], "glm-5.3")

    def test_qwen_config_is_not_taken_for_glm(self):
        self.assertFalse(resource_plan._is_glm53(ResourcePlanTests().config()))

    def test_layer_split_reads_the_declared_types(self):
        architecture = self.plan()["architecture"]
        self.assertEqual(architecture["linear_layers"], 34)
        self.assertEqual(architecture["full_layers"], 11)

    def test_kv_is_the_mla_latent_not_a_full_head_stack(self):
        state = self.plan()["state_bytes"]
        # 11 full layers x (512 + 0) latent x 4 bytes.
        self.assertEqual(state["kv_per_token_per_slot"], 11 * 512 * 4)

    def test_state_at_16k_is_small_enough_to_serve(self):
        # The whole point of MLA here: 16k of context costs well under a GiB.
        self.assertLess(self.plan()["state_bytes"]["total"], resource_plan.GIB)

    def test_container_is_measured_when_a_snapshot_is_given(self):
        container = self.plan()["container_bytes"]
        self.assertTrue(container["measured"])
        self.assertEqual(container["total"], 190 * resource_plan.GIB)
        # Routed experts stream; the remainder must stay resident.
        self.assertGreater(container["routed_experts"], 0)
        self.assertEqual(
            container["dense_core"],
            container["total"] - container["routed_experts"],
        )

    def test_routed_experts_cover_only_the_sparse_layers(self):
        # The first three layers are dense and hold no routed experts.
        routed_one = self.plan()["tier_plan"]["routed_expert_bytes"]
        self.assertEqual(
            self.plan()["container_bytes"]["routed_experts"],
            42 * 288 * routed_one,
        )

    def test_cache_below_topk_is_reported_as_a_fault(self):
        """A per-layer cache smaller than topk cannot hold one token's experts.

        Measured: at cap 6 against topk 8 the engine split every layer of every
        token into two blocks and evicted part of the block it had just read.
        Raising the budget so the cache holds topk took one review from 857 s
        to 437 s, so this is a fault to report, not a preference.
        """
        self.assertFalse(self.plan(ram_cache_gib=4.0)["checks"]["expert_cache_holds_topk"])
        self.assertTrue(self.plan(ram_cache_gib=14.0)["checks"]["expert_cache_holds_topk"])

    def test_ram_is_sized_by_resident_width_not_container_width(self):
        """The engine quantises on load, so RAM is not the file size.

        GLM-5.3-Flash stores 18.1 GiB of resident tensors as bf16 and holds
        9 GiB of them in the process. Sizing RAM off the container refused a
        configuration that fits and measured nearly twice as fast.
        """
        container_sized = self.plan(ram_cache_gib=14.0)
        measured = self.plan(ram_cache_gib=14.0,
                             resident_ram_bytes=9 * resource_plan.GIB)
        self.assertLess(
            measured["requirements_bytes"]["ram"],
            container_sized["requirements_bytes"]["ram"],
        )
        self.assertTrue(measured["checks"]["ram"])
        # The guard must still refuse a budget that genuinely does not fit.
        self.assertFalse(
            self.plan(ram_cache_gib=25.0,
                      resident_ram_bytes=9 * resource_plan.GIB)["checks"]["ram"]
        )

    def test_ram_check_fails_when_the_host_is_too_small(self):
        self.assertFalse(self.plan(system_ram_gib=4.0)["checks"]["ram"])

    def test_disk_check_fails_when_the_volume_is_too_small(self):
        self.assertFalse(self.plan(disk_free_gib=50.0)["checks"]["disk"])





if __name__ == "__main__":
    unittest.main()
