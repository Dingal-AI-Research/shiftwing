import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "audit_release",
    TOOLS / "audit_release.py",
)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def manifest(model: str) -> dict:
    required = audit.MODEL_REQUIREMENTS[model]
    return {
        **required,
        "complete": True,
        "xbits": "int4g128",
        "io_bits": 8,
        "shared_bits": 8,
        "group_size": 128,
        "include_mtp": False,
    }


def prefix_gate() -> dict:
    rows = [
        {
            "tf_matches": 62,
            "tf_compared": 64,
            "tf_agreement": 62 / 64,
        }
        for _ in range(20)
    ]
    return {
        "prompts": 20,
        "rows": rows,
        "tf_matching_tokens": 1240,
        "tf_compared_tokens": 1280,
        "tf_agreement": 1240 / 1280,
        "tf_prompts_at_85_percent": 20,
        "acceptance": {"passed": True},
    }


def ppl_gate(*, reference: bool) -> dict:
    value = {
        "ppl": 1.1,
        "tokens": 510,
        "acceptance": {"passed": True},
    }
    if reference:
        value.update(
            {
                "llama_ppl": 1.11,
                "relative_ppl_delta": (1.1 - 1.11) / 1.11,
            }
        )
    return value


def tool_gate() -> dict:
    return {
        "passed": True,
        "tool_call_response": {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 8},
        },
        "final_response": {
            "choices": [
                {
                    "message": {"content": "Paris is 18 °C."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 8},
        },
        "profile": {
            "resident": {
                "device_moe": 40,
                "host_moe": 0,
                "router_d2h_bytes": 1024,
                "logits_d2h_bytes": 1024,
            }
        },
    }


def tier_gate() -> dict:
    tiers = {
        "disk": 27840,
        "ram": 2104,
        "vram": 776,
        "vram_gb": 6.0,
    }
    return {
        "model_family": "ornith-1.0",
        "expert_lowbit_manifest": None,
        "telemetry_error": None,
        "configuration": {
            "warmup_passes": 2,
            "measured_passes": 1,
            "max_tokens": 64,
            "context": 4096,
            "expert_ram_gb": 18.0,
            "cuda_expert_gb": 6.0,
            "ram_headroom_gb": 1.0,
            "cuda_headroom_gb": 1.0,
            "minimum_tps": 0.85,
            "uring_persist": True,
            "pinned_upload": True,
            "decode_protect": True,
            "decode_protect_prewarm": True,
            "predictive_prefetch": False,
        },
        "measured": [
            {
                "text": f"coherent {index}",
                "stats": {
                    "completion_tokens": 64,
                    "tokens_per_second": 0.86,
                },
            }
            for index in range(4)
        ],
        "tiers": tiers,
        "emap": {"rows": 60, "cols": 512},
        "telemetry_summary": {
            "rows": 60,
            "cols": 512,
            "experts": 60 * 512,
            "tier_counts": tiers,
        },
        "hardware": {"gpus": 1, "vram_total_gb": 16.0},
        "resident": {
            "layers": 60,
            "device_moe": 60,
            "host_moe": 0,
            "activation_h2d_bytes": 1024,
            "activation_d2h_bytes": 1024,
            "router_d2h_bytes": 1024,
            "logits_d2h_bytes": 1024,
        },
        "acceptance": {"passed": True},
        "summary": {
            "automatic_gate_pass": True,
            "outputs_nonempty": True,
            "telemetry_complete": True,
            "cuda_active": True,
            "resident_cuda_graph": True,
            "throughput_pass": True,
            "sustained_tps": 0.86,
        },
    }


def doctor_gate() -> dict:
    expected = audit.MODEL_REQUIREMENTS["ornith397"]
    preregistered = {
        "source": expected["source"],
        "source_fingerprint": expected["source_fingerprint"],
        "source_shards": expected["source_shards"],
        "output_shards": expected["output_shards"],
        "logical_tensors": expected["logical_tensor_count"],
        "physical_tensors": expected["tensor_count"],
        "data_bytes": expected["data_bytes"],
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "checks": [
            {"id": name, "status": "pass"}
            for name in (
                "model.path",
                "model.config",
                "model.tokenizer",
                "model.quantization",
                "engine.binary",
                "accelerator.cuda",
                "runtime.state",
            )
        ]
        + [
            {
                "id": "model.container",
                "status": "pass",
                "details": {
                    "conversion_complete": True,
                    "shards": expected["output_shards"],
                    "parsed_shards": expected["output_shards"],
                    "header_tensors": expected["tensor_count"],
                    "tensor_bytes": expected["data_bytes"],
                },
            },
            {
                "id": "model.ledger",
                "status": "pass",
                "details": {
                    "source": expected["source"],
                    "source_fingerprint": expected["source_fingerprint"],
                    "committed_sources": expected["source_shards"],
                    "output_shards": expected["output_shards"],
                    "logical_tensors": expected["logical_tensor_count"],
                    "physical_tensors": expected["tensor_count"],
                    "data_bytes": expected["data_bytes"],
                    "hashes_requested": True,
                    "hashes_checked": expected["output_shards"],
                    "preregistered_expectations": preregistered,
                },
            },
            {
                "id": "runtime.resources",
                "status": "pass",
                "details": {
                    "checks": {"disk": True, "ram": True, "vram": True}
                },
            },
        ],
        "plan": {
            "checks": {"disk": True, "ram": True, "vram": True},
            "safe": True,
        },
    }


class ReleaseAuditTests(unittest.TestCase):
    def _workspace(self, root: Path) -> None:
        cdir = root / "c"
        cdir.mkdir()
        manifests = {}
        for model in ("ornith35", "ornith397"):
            model_dir = cdir / model
            model_dir.mkdir()
            raw = manifest(model)
            (model_dir / "quantization.json").write_text(json.dumps(raw))
            manifests[model] = {
                name: raw.get(name) for name in audit.MANIFEST_FIELDS
            }

        artifacts = {
            "ornith35_prefix_gate.json": (
                "ornith35",
                prefix_gate(),
            ),
            "ornith35_ppl_gate.json": (
                "ornith35",
                ppl_gate(reference=True),
            ),
            "ornith35_tool_gate.json": ("ornith35", tool_gate()),
            "ornith397_ppl_smoke.json": (
                "ornith397",
                ppl_gate(reference=False),
            ),
            "ornith397_qualification.json": (
                "ornith397",
                tier_gate(),
            ),
            "ornith397_tool_qualification.json": (
                "ornith397",
                tool_gate(),
            ),
        }
        for model in ("ornith35", "ornith397"):
            source = audit.MODEL_REQUIREMENTS[model]["source"]
            artifacts[f"{model}_batch_abba.json"] = (
                model,
                {
                    "model_family": "ornith-1.0",
                    "prompt": {
                        "user_text": "Reply with exactly: colib batch ready",
                        "expected_exact_content": "colib batch ready",
                        "raw": False,
                    },
                    "speedups": {
                        "ab": 0.98,
                        "ba": 1.03,
                        "minimum": 0.98,
                        "geometric_mean": (0.98 * 1.03) ** 0.5,
                    },
                    "acceptance": {
                        "passed": True,
                        "expected_family": "ornith-1.0",
                        "expected_source": source,
                        "minimum_each": 0.95,
                        "minimum_geometric_mean": 1.0,
                    },
                },
            )
            artifacts[f"{model}_cancel_gate.json"] = (
                model,
                {
                    "model_family": "ornith-1.0",
                    "cuda": True,
                    "prompt": {
                        "user_text": "Reply briefly: cancellation probe",
                        "raw": False,
                    },
                    "tier_configuration": {
                        "warmup_passes": 2,
                        "trials": 20,
                        "context": 4096,
                        "ram_gb": 8.0 if model == "ornith35" else 18.0,
                        "ram_headroom_gb": 1.0,
                        "cuda_expert_gb": 6.0 if model == "ornith35" else 5.0,
                        "cuda_headroom_gb": 1.0,
                        "tiered_io": True,
                        "predictive_prefetch": False,
                        "uring_persist": False,
                        "pinned_upload": False,
                        "decode_protect": False,
                        "decode_protect_prewarm": False,
                        "expert_lowbit": None,
                    },
                    "trials": [
                        {
                            "cancelled": {
                                "pieces_before_cancel": 1,
                                "cancel_ack_s": 0.5,
                            },
                            "peer": {
                                "text": "peer",
                                "stats": {"completion_tokens": 8},
                            },
                            "slot_reuse": {"text": "reuse"},
                        }
                        for _ in range(20)
                    ],
                    "summary": {
                        "cancel_ack_samples_s": [0.5] * 20,
                        "p95_method": "nearest-rank",
                        "p95_cancel_ack_s": 0.5,
                    },
                    "tiers": {"vram": 64},
                    "hardware": {"gpus": 1, "vram_total_gb": 16.0},
                    "resident": {
                        "layers": 80,
                        "device_moe": 80,
                        "host_moe": 0,
                        "router_d2h_bytes": 1024,
                        "logits_d2h_bytes": 1024,
                    },
                    "acceptance": {
                        "passed": True,
                        "maximum_cancel_ack_s": (
                            1.0 if model == "ornith35" else 3.0
                        ),
                        "measured_statistic": "p95_cancel_ack_s",
                    },
                },
            )
            artifacts[f"{model}_web_gate.json"] = (
                model,
                {
                    "model_family": "ornith-1.0",
                    "cuda": True,
                    "web_bundle": True,
                    "configuration": {
                        "expected_exact_content": "colib ready",
                        "requests": 2,
                        "kv_slots": 2,
                        "max_tokens": 16,
                        "context": 4096,
                        "ram_gb": 8.0 if model == "ornith35" else 18.0,
                        "ram_headroom_gb": 1.0,
                        "cuda_expert_gb": 6.0 if model == "ornith35" else 5.0,
                        "cuda_headroom_gb": 1.0,
                        "predictive_prefetch": False,
                        "uring_persist": False,
                        "pinned_upload": False,
                        "decode_protect": False,
                        "decode_protect_prewarm": False,
                        "expert_lowbit": None,
                    },
                    "requests": [
                        {
                            "content": "colib ready",
                            "usage": {"completion_tokens": 3},
                        },
                        {
                            "content": "colib ready",
                            "usage": {"completion_tokens": 3},
                        },
                    ],
                    "profile": {
                        "resident": {
                            "layers": 80,
                            "device_moe": 80,
                            "host_moe": 0,
                            "router_d2h_bytes": 1024,
                            "logits_d2h_bytes": 1024,
                        }
                    },
                    "final_health": {
                        "scheduler": {
                            "active": 0,
                            "queued": 0,
                            "completed": 2,
                        },
                        "hwinfo": {"gpus": 1, "vram_total_gb": 16.0},
                        "tiers": {"vram": 64},
                    },
                    "acceptance": {"passed": True},
                },
            )
        for filename, (model, body) in artifacts.items():
            body["model_manifest"] = manifests[model]
            (cdir / filename).write_text(json.dumps(body))
        (cdir / "ornith397_doctor.json").write_text(json.dumps(doctor_gate()))

    def _q3_workspace(self, root: Path) -> None:
        self._workspace(root)
        cdir = root / "c"
        for index, relative in enumerate(audit.ENGINE_SOURCE_FILES):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"engine-source-{index}\n".encode())
        model = cdir / "ornith397"
        q3_raw = {
            **audit.Q3_REQUIREMENTS,
            "data_bytes": 157_000_000_000,
            "files": [{} for _ in range(480)],
        }
        q3_path = model / "expert-q3.json"
        q3_path.write_text(json.dumps(q3_raw))
        q3_identity = {
            name: q3_raw.get(name) for name in audit.Q3_ARTIFACT_FIELDS
        }
        base_raw = json.loads((model / "quantization.json").read_text())
        base_identity = {
            name: base_raw.get(name) for name in audit.MANIFEST_FIELDS
        }

        def bind(body: dict) -> dict:
            body["model_manifest"] = base_identity
            body["expert_lowbit_manifest"] = q3_identity
            return body

        prefix = bind(prefix_gate())
        tier = bind(tier_gate())
        tier["configuration"].update(
            {
                "expert_q2": False,
                "expert_q3": True,
                "q3_native": False,
                "minimum_tps": audit.ORNITH397_Q3_MINIMUM_TPS,
            }
        )
        coherence = json.loads(json.dumps(tier))
        coherence["configuration"].update(
            {"warmup_passes": 0, "minimum_tps": 0.000001}
        )
        tools = bind(tool_gate())
        q3_artifacts = {
            "ornith397_q3_prefix_gate.json": prefix,
            "ornith397_q3_coherence.json": coherence,
            "ornith397_q3_qualification.json": tier,
            "ornith397_q3_tool_gate.json": tools,
            "ornith397_int4_prefix_reference.json": {
                "acceptance": {"passed": True}
            },
        }
        for filename, body in q3_artifacts.items():
            (cdir / filename).write_text(json.dumps(body))

        q3_doctor = {
            "schema_version": 1,
            "status": "ok",
            "passed": True,
            "bits": 3,
            "manifest_sha256": audit._sha256(q3_path),
            "source_identity": q3_raw["signature"]["source_identity"],
            "layers": 60,
            "experts_per_layer": 512,
            "group_size": 128,
            "iterations": 3,
            "experts_per_file": 64,
            "file_count": 480,
            "tensor_count": 276480,
            "data_bytes": q3_raw["data_bytes"],
            "headers_verified": 480,
            "hashes_verified": 480,
        }
        (cdir / "ornith397_q3_doctor.json").write_text(json.dumps(q3_doctor))

        for filename in (
            "ornith397_batch_abba.json",
            "ornith397_cancel_gate.json",
            "ornith397_web_gate.json",
        ):
            path = cdir / filename
            value = json.loads(path.read_text())
            value["expert_lowbit_manifest"] = q3_identity
            if filename == "ornith397_cancel_gate.json":
                value["tier_configuration"]["expert_lowbit"] = "int3g128"
            if filename == "ornith397_web_gate.json":
                value["configuration"]["expert_lowbit"] = "int3g128"
            path.write_text(json.dumps(value))

        engine = cdir / "qwen"
        engine.write_bytes(b"cuda-engine")
        state_path = cdir / "bench" / "ornith397_q3_gate8_pipeline.json"
        state_path.parent.mkdir()
        step_artifacts = {
            "q3_doctor": cdir / "ornith397_q3_doctor.json",
            "int4_reference": cdir / "ornith397_int4_prefix_reference.json",
            "q3_teacher_forced": cdir / "ornith397_q3_prefix_gate.json",
            "q3_coherence": cdir / "ornith397_q3_coherence.json",
            "q3_tools": cdir / "ornith397_q3_tool_gate.json",
            "q3_tier": cdir / "ornith397_q3_qualification.json",
        }
        state = {
            "schema_version": 2,
            "status": "passed",
            "coherence_review_acknowledged": True,
            "q3_ppl_waived": True,
            "q3_minimum_tps": audit.ORNITH397_Q3_MINIMUM_TPS,
            "q3_policy_basis": audit.ORNITH397_Q3_POLICY_BASIS,
            "base_manifest_sha256": audit._sha256(model / "quantization.json"),
            "q3_manifest_sha256": audit._sha256(q3_path),
            "int4_ppl_baseline_sha256": audit._sha256(
                cdir / "ornith397_ppl_smoke.json"
            ),
            "engine_sha256": audit._sha256(engine),
            "engine_source_sha256": audit.engine_source_sha256(root),
            "steps": {
                name: {
                    "status": "passed",
                    "argv": ["fixture"],
                    "artifact": str(path.resolve()),
                    "artifact_sha256": audit._sha256(path),
                }
                for name, path in step_artifacts.items()
            },
        }
        state_path.write_text(json.dumps(state))

    def test_complete_manifest_bound_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            result = audit.audit_release(root)
        self.assertTrue(result["summary"]["passed"])
        self.assertEqual(result["summary"]["check_count"], 15)

    def test_complete_q3_manifest_bound_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._q3_workspace(root)
            result = audit.audit_release(root, ornith397_expert_q3=True)
        self.assertTrue(result["summary"]["passed"], result["summary"])
        self.assertEqual(result["summary"]["check_count"], 20)

    def test_q3_release_rejects_mixed_gate9_sidecar_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._q3_workspace(root)
            path = root / "c" / "ornith397_web_gate.json"
            value = json.loads(path.read_text())
            value["expert_lowbit_manifest"]["data_bytes"] += 1
            path.write_text(json.dumps(value))
            result = audit.audit_release(root, ornith397_expert_q3=True)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.web", result["summary"]["failures"])

    def test_q3_release_rejects_missing_ppl_waiver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._q3_workspace(root)
            path = root / "c" / "bench" / "ornith397_q3_gate8_pipeline.json"
            value = json.loads(path.read_text())
            value["q3_ppl_waived"] = False
            path.write_text(json.dumps(value))
            result = audit.audit_release(root, ornith397_expert_q3=True)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.ppl", result["summary"]["failures"])

    def test_q3_release_rejects_engine_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._q3_workspace(root)
            path = root / audit.ENGINE_SOURCE_FILES[1]
            path.write_bytes(path.read_bytes() + b"drift\n")
            result = audit.audit_release(root, ornith397_expert_q3=True)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.coherence_review", result["summary"]["failures"])

    def test_atomic_report_publication_replaces_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "release_audit.json"
            audit._atomic_write_text(path, "first\n")
            audit._atomic_write_text(path, "second\n")
            self.assertEqual(path.read_text(), "second\n")
            self.assertEqual(
                list(path.parent.glob(".release_audit.json.tmp-*")),
                [],
            )

    def test_wrong_artifact_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            path = root / "c" / "ornith397_web_gate.json"
            value = json.loads(path.read_text())
            value["model_manifest"]["source_fingerprint"] = "wrong"
            path.write_text(json.dumps(value))
            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.web", result["summary"]["failures"])

    def test_wrong_container_cardinality_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            path = root / "c" / "ornith397" / "quantization.json"
            value = json.loads(path.read_text())
            value["output_shards"] -= 1
            value["tensor_count"] -= 1
            value["io_bits"] = 4
            path.write_text(json.dumps(value))
            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.manifest", result["summary"]["failures"])

    def test_forged_doctor_hash_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            path = root / "c" / "ornith397_doctor.json"
            value = json.loads(path.read_text())
            ledger = next(
                item for item in value["checks"] if item["id"] == "model.ledger"
            )
            ledger["details"]["hashes_checked"] = 121
            path.write_text(json.dumps(value))
            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.doctor", result["summary"]["failures"])

    def test_ornith397_revised_threshold_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            path = root / "c" / "ornith397_qualification.json"
            value = json.loads(path.read_text())
            for row in value["measured"]:
                row["stats"]["tokens_per_second"] = 0.849
            value["summary"]["sustained_tps"] = 0.849
            path.write_text(json.dumps(value))
            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.tier", result["summary"]["failures"])

    def test_forged_numerical_and_tool_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)

            prefix_path = root / "c" / "ornith35_prefix_gate.json"
            prefix = json.loads(prefix_path.read_text())
            prefix["rows"][0]["tf_matches"] = 1
            prefix_path.write_text(json.dumps(prefix))

            ppl_path = root / "c" / "ornith35_ppl_gate.json"
            ppl = json.loads(ppl_path.read_text())
            ppl["relative_ppl_delta"] = 0.0
            ppl_path.write_text(json.dumps(ppl))

            tool_path = root / "c" / "ornith397_tool_qualification.json"
            tool = json.loads(tool_path.read_text())
            tool["tool_call_response"]["choices"][0]["message"]["tool_calls"][0][
                "function"
            ]["arguments"] = '{"city":"London"}'
            tool_path.write_text(json.dumps(tool))

            tier_path = root / "c" / "ornith397_qualification.json"
            tier = json.loads(tier_path.read_text())
            tier["resident"]["host_moe"] = 1
            tier_path.write_text(json.dumps(tier))

            batch_path = root / "c" / "ornith35_batch_abba.json"
            batch = json.loads(batch_path.read_text())
            batch["speedups"]["ab"] = 0.5
            batch_path.write_text(json.dumps(batch))

            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith35.prefix", result["summary"]["failures"])
        self.assertIn("ornith35.ppl", result["summary"]["failures"])
        self.assertIn("ornith397.tools", result["summary"]["failures"])
        self.assertIn("ornith397.tier", result["summary"]["failures"])
        self.assertIn("ornith35.batch", result["summary"]["failures"])

    def test_forged_cancellation_p95_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            path = root / "c" / "ornith397_cancel_gate.json"
            value = json.loads(path.read_text())
            value["summary"]["p95_cancel_ack_s"] = 0.1
            path.write_text(json.dumps(value))
            result = audit.audit_release(root)
        self.assertFalse(result["summary"]["passed"])
        self.assertIn("ornith397.cancel", result["summary"]["failures"])


if __name__ == "__main__":
    unittest.main()
