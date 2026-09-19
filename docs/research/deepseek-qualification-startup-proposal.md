# Proposed native DeepSeek startup qualification check

This patch is **not applied**. Automatic approval review required explicit permission for changing production startup. It changes only `c/deepseek_v4.c` and the command-line entry point of `c/tools/deepseek_v4_qualification.py`.

The permanent startup lock becomes an identity-bound qualification check. Startup still exits 78 if the report is absent, incomplete, stale, has changed evidence, lacks five complete planted-defect reviews (including actual 32768 input tokens), exceeds the 20-minute warm limit, or violates measured memory limits. It requires the tested CUDA reviewer profile. Validation occurs before loading dense weights or allocating GPU caches.

The existing explicit experimental path remains available for standalone qualification. The patch does not create a passing report, activate the reviewer, edit live settings, start/restart services, or change other engines. Native doctor routing is a separate pending change.

The standalone validator has ten passing fault tests. The proposed C integration still needs compilation and integration tests after approval. A final qualification report must be generated against that compiled executable before ordinary serving can work.
