# Graph Report - .  (2026-08-13)

## Corpus Check
- 4 files · ~23,688 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 125 nodes · 263 edges · 7 communities
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Web Request Path and Orientations
- Bedrock Generator and Parity
- CLI and Output Path
- Vector Harness and Divergences
- Sampling and Test Doctrine
- CUDA Search Kernel
- Pattern Editor and Match Estimate

## God Nodes (most connected - your core abstractions)
1. `bd_classify()` - 12 edges
2. `bd_probe()` - 12 edges
3. `main()` - 11 edges
4. `main()` - 10 edges
5. `parse_search()` - 10 edges
6. `render_view()` - 9 edges
7. `die()` - 8 edges
8. `derive()` - 8 edges
9. `gpu_search()` - 8 edges
10. `Strictness is pinned by vectors, not by fp_proof` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Nether roof band is unverified against vanilla` --references--> `bd_classify()`  [EXTRACTED]
  CONTRIBUTING.md → bedrock.h
- `CUDA is not always the fastest path` --references--> `cpu_search()`  [EXTRACTED]
  README.md → main.c
- `-ffp-contract=off / --fmad=false, never --use_fast_math` --rationale_for--> `search_kernel()`  [INFERRED]
  CONTRIBUTING.md → search.cu
- `Half-typed input is not an instruction` --semantically_similar_to--> `as_int()`  [INFERRED] [semantically similar]
  web/index.html → web/serve.py
- `bd_hash 32-bit x vs 64-bit z asymmetry` --references--> `main()`  [INFERRED]
  CONTRIBUTING.md → tools/check.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Five generator internals that diverge silently if simplified** — contributing_bd_hash_asymmetry, contributing_nextfloat_single_precision, contributing_float_narrowing_licence, contributing_comparison_strictness, contributing_two_derivers, contributing_fp_contract_flags [EXTRACTED 1.00]
- **What makes the sampled estimate honest** — readme_measured_estimate, readme_layers_not_independent, readme_two_stage_sampling, readme_z_banded_match_rate [EXTRACTED 1.00]
- **Controls that bound one web request** — readme_area_cap, readme_request_timeout, readme_cancel_search, readme_measured_estimate [INFERRED 0.85]

## Communities (7 total, 0 thin omitted)

### Community 0 - "Web Request Path and Orientations"
Cohesion: 0.11
Nodes (29): No C code in the serving path, Exception, Search area cap guards against a stray zero, Cancel kills the binary, not just the connection, Timeout is per request, not per scan, Scan more than one orientation, Half-typed input is not an instruction, Self-mapping orientations are dropped, not scanned twice (+21 more)

### Community 1 - "Bedrock Generator and Parity"
Cohesion: 0.20
Nodes (24): BD_FN, bd_check(), bd_classify(), bd_hash(), bd_lerp(), bd_lerp_from_progress(), bd_lerp_progress(), bd_next() (+16 more)

### Community 2 - "CLI and Output Path"
Cohesion: 0.20
Nodes (21): add_block(), bd_add_match(), be64(), bd_derivers, bd_match, cpu_search(), derive(), die() (+13 more)

### Community 3 - "Vector Harness and Divergences"
Cohesion: 0.26
Nodes (12): Deliberate divergences ledger, Missing pattern/ directory matches every column, Nether roof band is unverified against vanilla, Recorded vectors replace the Java cross-check, The two derivers are different entry points, Known limits: caps reported, never silently truncated, digest(), main() (+4 more)

### Community 4 - "Sampling and Test Doctrine"
Cohesion: 0.20
Nodes (11): bd_hash 32-bit x vs 64-bit z asymmetry, Measure performance, do not reason about it, A test that cannot fail is not a test, Match rates band by Z, so the sample spans every Z, main(), make_handler(), _png_pixels(), A slice spanning the whole Z range of the search. Match rates are not spatially… (+3 more)

### Community 5 - "CUDA Search Kernel"
Cohesion: 0.22
Nodes (10): -ffp-contract=off / --fmad=false, never --use_fast_math, cudaError_t, __global__, CUDA is not always the fastest path, bd_block, bd_derivers, bd_match, fail() (+2 more)

### Community 6 - "Pattern Editor and Match Estimate"
Cohesion: 0.20
Nodes (10): Bedrock stacks vertically more than chance, Measured estimate instead of a probability model, Two-stage sampling keeps output bounded, Web GUI for drawing patterns, One fluid rem unit drives the whole layout, localStorage autosave of the whole editor state, Layers store a full MAXW grid regardless of displayed size, Pattern grid painting (three-state cells) (+2 more)

## Knowledge Gaps
- **2 isolated node(s):** `localStorage autosave of the whole editor state`, `Two-stage sampling keeps output bounded`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `parse_search()` connect `Web Request Path and Orientations` to `Vector Harness and Divergences`, `Pattern Editor and Match Estimate`?**
  _High betweenness centrality (0.238) - this node is a cross-community bridge._
- **Why does `Missing pattern/ directory matches every column` connect `Vector Harness and Divergences` to `Web Request Path and Orientations`, `CLI and Output Path`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Why does `The One Rule: generator must match Minecraft` connect `Bedrock Generator and Parity` to `Web Request Path and Orientations`, `Vector Harness and Divergences`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `bd_classify()` (e.g. with `Strictness is pinned by vectors, not by fp_proof` and `prefilter()`) actually correct?**
  _`bd_classify()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `main()` (e.g. with `bd_hash 32-bit x vs 64-bit z asymmetry` and `A test that cannot fail is not a test`) actually correct?**
  _`main()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `localStorage autosave of the whole editor state`, `Two-stage sampling keeps output bounded` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Web Request Path and Orientations` be split into smaller, more focused modules?**
  _Cohesion score 0.11397849462365592 - nodes in this community are weakly interconnected._