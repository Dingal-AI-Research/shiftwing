# Phase 11 Experiment 4: Pinned source fetch completion

**Date:** 2026-08-16
**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
**Revision:** `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
**Repository commit:** `35d9ab86fa2b8f8acbf56c8438413dfb08f2a2a7` (dirty worktree)

## Commands

```bash
./.venv/bin/python c/tools/fetch_deepseek_v4.py \
  --output c/.deepseek-v4-flash-0731.source --workers 4

./.venv/bin/python c/tools/inventory_deepseek_v4.py \
  --source c/.deepseek-v4-flash-0731.source --root . \
  --output c/bench/deepseek_v4_inventory_complete.json
```

The fetch command is stored verbatim in each attempt entry. The inventory JSON
stores its command, creation time, repository commit/branch/dirty state, source
identity hashes, checkpoint counts, layout analysis, and terminal acceptance.

## Resumption evidence

The hash-bound fetch ledger contains three attempts:

| attempt | start UTC | durable before/after | terminal state |
|---:|---|---:|---|
| 1 | 2026-08-16T11:29:34.111949+00:00 | 49 / 58 | interrupted; stale running attempt recovered |
| 2 | 2026-08-16T14:06:31.307520+00:00 | 58 / 58 | interrupted; stale running attempt recovered |
| 3 | 2026-08-16T15:18:34.854404+00:00 | 58 / 62 | complete at 2026-08-16T15:47:27.494777+00:00 |

Attempt 3 first rehashed the complete 160.5 GB durable corpus, then resumed the
four partial ranges. Shards 44, 46, 47, and 48 committed atomically in order.
No `.model-*.partial` file remains.

## Final source identity

- source directory bytes: 166,898,734,250;
- completed files: 62;
- weight shards: 48;
- source marker status: `complete`;
- API SHA-256:
  `8c6058c97926c21e56e226d57468f2aef5af647aad4a9a571139d41828260aa2`;
- fetch-state SHA-256:
  `7ccc03191e8258a3ecce81af5a9a7c1a7fab1a648b662d67865c2842ea2f224c`;
- source-marker SHA-256:
  `e55e5a9031c57ebba4230bbbab6398154da4eeb8bf869ab2cf4a0317f466a4a0`;
- config SHA-256:
  `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`;
- index SHA-256:
  `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.

## Independent inventory

A separate single-threaded pass rehashed every completed file and accepted:

- 72,317 indexed tensors;
- 48 weight shards;
- 166,878,536,440 indexed payload bytes;
- 296,138,115-byte worst-case 4 KiB alignment padding;
- 167,174,674,555-byte native-container payload upper bound;
- 1,564 base-dense records;
- 66,048 base routed-expert records;
- 97 DSpark-dense records;
- 4,608 DSpark routed-expert records;
- 43 base layers, each with exactly 256 complete experts;
- three bundled DSpark expert layers;
- no failures.

## Storage disposition

After fetch, ext4 has 413,175,259,136 bytes available. The VHDX grew from
611,744,481,280 to 618,220,486,656 bytes while completing the four partial
shards; Windows `C:\` has 47,417,536,512 bytes free. This is enough to retain
the complete source but not enough to authorize the 167 GB conversion while
preserving the 100 GiB post-conversion floor.

Ornith397 therefore remains protected for its fresh control. The next storage
decision is made only after that control and its reconstruction evidence are
independently recorded.

## Acceptance

**PASS:** the pinned source fetch and independent full-file/tensor inventory are
complete and resumable. Gate 11.0 remains open only for an interrupted full
real conversion/resume proof.
