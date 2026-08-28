# MTP-ASSEMBLY-SPEC.md
## Qwen3.8-Flash-Next (arch `qwen4_exp`) MTP head — definitive assembly spec for the MLX port

Status target: real-weight draft acceptance is currently ZERO with the speculative
machinery proven exact, i.e. the head forward is mis-assembled. This document is the
line-by-line reference to diff against.

Sources reconciled: llama.cpp PR #27739 (JJJYmmm fork @ dfa0c0f, convert-time fused
`eh_proj` variant), llama.cpp PR #27742 (no MTP yet; qwen35 `graph_mtp` template),
vLLM PR #53896 (`vllm/models/qwen4_exp/nvidia/mtp.py`), TokenSpeed branch
`zt/qwen3.8_next` (`qwen4_exp_nextn.py`), SGLang PR #36497 (`qwen4_exp_mtp.py`,
`hyperconnection.py`), HF checkpoint tensor index, RadixArk NVFP4 config.json.
All four MTP implementations agree on the math; the two representational
disagreements are resolved in §1.7.

---

## 0. Notation, dims, tensor inventory

```
H  = 2560            hidden_size
S  = 4               hc_count (hyper-connection streams)
D  = S*H = 10240     flattened stream width
R  = 320             hc_lowrank
V  = 248320          vocab
T                    tokens in the batch
eps                  config rms_norm_eps (reuse the trunk's value)
theta_mtp = 1e7      rope theta of the MTP block (config text_config.mtp.rope_theta)
```

**Flatten convention (load-bearing):** every width-D vector is the row-major flatten
of `[S, H]` — element index `s*H + h`, stream index OUTER, hidden index INNER.
`x.reshape(T, S, H)` must recover the streams. All `[·,10240]` and `[10240]`
checkpoint tensors assume this layout (SGLang: `hidden.view(T, hc_count, hidden_size)`
directly after the flat norm; `unflatten(-1, (hc, H))` in TokenSpeed).

**Ground-truth checkpoint inventory (31 `mtp.*` tensors) and where each is consumed:**

| tensor | shape | consumed at (spec step) |
|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | [2560] | §1.2 step F2 (norm of token embedding) |
| `mtp.fc_embedding.weight` | [2560,2560] | §1.2 step F3 |
| `mtp.pre_fc_norm_hidden.weight` | [10240] | §1.2 step F4 (ONE variance over all 10240) |
| `mtp.fc_hidden.weight` | [2560,2560] | §1.2 step F5 (shared across the 4 streams) |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | [10240] | §1.4 A1 (grouped, per-stream variance) |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down` | [320,10240] | §1.4 A1 |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up` | [10240,320] | §1.4 A1 |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight` | [4,10240] | §1.4 A1 (inject gate) / A3 (combine) |
| `mtp.layers.0.self_attn.q_proj` (fused q+output-gate, 2× width) | — | §1.4 A2 |
| `mtp.layers.0.self_attn.{k_proj,v_proj,o_proj,q_norm,k_norm}` | — | §1.4 A2 |
| `mtp.layers.0.self_attn.indexer.*` (index_qk_proj, q/k layernorm) | — | **UNUSED in the dense baseline** (§1.7-D3) |
| `mtp.layers.0.mlp_hyper_connection.{hc_norm,down,up,block_inject}` | as attn | §1.4 M1/M3 |
| `mtp.layers.0.mlp.gate` (router), `experts.{gate_up_proj,down_proj}` (512) | — | §1.4 M2 |
| `mtp.layers.0.mlp.shared_expert.*`, `shared_expert_gate` | — | §1.4 M2 |
| `mtp.hyper_connection_mixer.hc_norm.weight` | [10240] | §1.5 step O2 (grouped, per-stream variance) |
| `mtp.hyper_connection_mixer.input_mix_weight_down` | [320,10240] | §1.5 O2 |
| `mtp.hyper_connection_mixer.input_mix_weight_up` | [10240,320] | §1.5 O2 |

**Shared with the trunk (no mtp copies exist in the checkpoint):**
`model.embed_tokens.weight` (token embedding lookup, §1.1), `lm_head.weight`
(§1.5 O4). There is **no** `mtp.norm`, **no** `mtp.shared_head.*`, **no**
`mtp.embed_tokens` (§1.7-D2). The trunk's final norm / final mixer are NOT used
inside the head.

Every listed tensor is consumed exactly once per head forward, except the QSA
indexer tensors, which are deliberately unused (flagged, §1.7-D3). If your
implementation consumes any tensor zero times or twice (e.g. reuses
`pre_fc_norm_hidden` as a final norm), it is wrong by construction.

---

## 1. The forward pass (one head step)

Signature (per token; batched over T):

```
head(token_id: int, h_prev: float[D], pos: int, kv_cache) -> (logits: float[V], m_out: float[D])
```

- `h_prev` is the FLATTENED 4-STREAM HYPER-CONNECTION STATE `[D]=[10240]` of the
  *previous* position — the trunk's pre-final-mixer state on step 0, the head's own
  `m_out` on steps ≥ 1. It is NEVER the collapsed/post-mixer `[2560]` hidden.
- `m_out` is the head's own pre-final-mixer stream state at `pos`, fed to the next step.

### 1.0 Primitives

**GemmaRMSNorm(x, w, width)** — variance over the last `width` dims, zero-centered
weight:

```
y = x * rsqrt(mean(x^2, axis=-1 over width) + eps) * (1 + w)
```

The `(1 + w)` is Gemma-style (SGLang/vLLM both instantiate `GemmaRMSNorm`;
llama.cpp bakes the +1 in at convert time). All norms in this spec use it:
`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, every `hc_norm`, `q_norm`, `k_norm`.
Sanity: checkpoint norm weights should cluster near 0, not near 1 (§4 T2c).

**GroupedRMSNorm(X, w)** for stream states, `X: [T,S,H]`, `w: [D]` viewed `[S,H]`:
variance computed PER (token, stream) over H only:

```
Xn[t,s,:] = X[t,s,:] * rsqrt(mean(X[t,s,:]^2) + eps) * (1 + w[s,:])
```

**GatedResidual.mix(X, hc_norm, Wdown, Wup [, Winject])** — the hyper-connection
pre-mixer. `X: [T,S,H]`; returns `(mixed: [T,H], Xn_flat: [T,D] [, gamma: [T,S]])`:

```
1. Xn      = GroupedRMSNorm(X, hc_norm)                # per-stream variance
2. xn_flat = Xn.reshape(T, D)                          # stream-major flatten
3. z       = silu( (xn_flat @ Wdown.T) / S )           # [T,R]; divide by S BEFORE silu
4. g       = sigmoid( z @ Wup.T ).reshape(T, S, H)     # [T,S,H]
5. mixed   = mean over s of ( g[:,s,:] * Xn[:,s,:] )   # [T,H]; MEAN, not sum
6. (if Winject given)
   gamma   = 2 * sigmoid( (xn_flat @ Winject.T) / S )  # [T,S]; from the NORMED flat
```

At init `gamma == 1` (2·σ(0)) — the ×2 and the two ÷S scalings are trained-in;
omitting any of them mis-scales every residual.

**GatedResidual.combine(X, y, gamma)** — the post-mixer injection. `y: [T,H]` is the
sub-block output; adds into the RAW (pre-norm) streams:

```
X'[t,s,:] = X[t,s,:] + gamma[t,s] * y[t,:]             # broadcast y to all S streams
```

(vLLM defers this combine into the next mix's norm — "delayed combine" — which is
algebraically identical; implement it immediately as above.)

### 1.1 Inputs

```
I1. e_tok = embed_tokens[token_id]          # [H], the TRUNK's model.embed_tokens
I2. h     = h_prev                          # [D], see §2 for which row this must be
```

Do NOT run the trunk's "replicate the embedding into S streams" input path here —
that is only for trunk layer 0. The head builds its stream state via the fusion
below.

### 1.2 Input fusion — `residual_linear_shared` (ADD, not concat)

```
F1. (nothing precedes; h is used raw, un-normed, as the fusion input)
F2. e_n  = GemmaRMSNorm(e_tok, pre_fc_norm_embedding, width=H)      # [H]
F3. e_p  = e_n @ fc_embedding.T                                     # [H]
F4. h_n  = GemmaRMSNorm(h, pre_fc_norm_hidden, width=D)             # [D]
         # ONE variance over the full 10240 vector (all 4 streams jointly),
         # elementwise (1+w) with w=[10240]. NOT per-stream. (llama.cpp comment:
         # "norms the whole hyper-connection row ... unlike deepseek4".)
F5. Hs   = h_n.reshape(S, H) @ fc_hidden.T                          # [S,H]
         # ONE [2560,2560] weight applied to EACH of the 4 normed streams.
F6. X0   = Hs + e_p[None, :]                                        # [S,H]
         # the projected embedding is broadcast-ADDED to EVERY stream.
```

`X0: [T,S,H]` is the head's stream state entering the decoder layer.
Equivalence note: llama.cpp PR #27739 fuses F3+F5+F6 at CONVERT time into one
`eh_proj = [fc_embedding | fc_hidden]` `[2H,H]` matmul of `concat(e_n, h_n_stream)`
per stream — `A·e + B·h == [A|B]·concat(e,h)`. The checkpoint ships the two `[H,H]`
weights, so at runtime you implement the ADD form above. There is NO DeepSeek-style
`Linear(2H,H)` over `concat(norm(e), norm(h))` in this model.

### 1.3 (reserved — no step between fusion and the layer; no norm, no collapse)

### 1.4 Decoder layer — one full trunk-style block at stream width

This is byte-for-byte a trunk layer (`mtp.layers.0` ≙ layer index 48), forced to:
`layer_type = full_attention`, PLE OFF (`ple_layer_ids=[]`), `attn_output_gate=True`,
`rope_theta = theta_mtp = 1e7`, its OWN KV cache. **Reuse your proven trunk-layer
code**; only the config overrides above differ.

Attention half:

```
A1. mixed, Xn_flat, gamma_a = mix(X0, attn_hc.hc_norm, attn_hc.down, attn_hc.up,
                                  attn_hc.block_inject)             # mixed: [T,H]
A2. y_a = SelfAttention(mixed, pos, kv_cache):
      - q_proj is FUSED [q | output-gate]: out-width = 2 * n_head * head_dim,
        interleaved PER HEAD (head h occupies rows [h*2*hd, (h+1)*2*hd): first hd
        rows = Q, next hd = gate) — take strided views, do not de-interleave weights.
      - q = per-head GemmaRMSNorm(q, q_norm); k = per-head GemmaRMSNorm(k, k_norm)
      - RoPE(pos) with theta_mtp=1e7 (m-rope sections as in the trunk; for text-only
        this reduces to standard rope) applied to q and k
      - DENSE causal attention against this head's own KV cache (indexer skipped)
      - out = attn_out * sigmoid(gate)     # elementwise, before o_proj
      - y_a = out @ o_proj.T
A3. X1 = combine(X0, y_a, gamma_a)                                  # [T,S,H]
```

MoE half:

```
M1. mixed, _, gamma_m = mix(X1, mlp_hc.hc_norm, mlp_hc.down, mlp_hc.up,
                            mlp_hc.block_inject)                    # [T,H]
M2. y_m = MoE(mixed)   # identical to the trunk MoE: router `gate` softmax top-k over
        # 512 experts (same top-k / routed scaling / normalization as your trunk),
        # PLUS shared expert: y_m = moe_out + shexp(mixed) * sigmoid(mixed @ shared_expert_gate.T)
M3. X2 = combine(X1, y_m, gamma_m)                                  # [T,S,H]
```

### 1.5 Output: feedback tap, final mixer, logits

```
O1. m_out = X2.reshape(T, D)          # FEEDBACK for the next draft step — captured
                                      # BEFORE the final mixer, for EVERY token
                                      # (before any output-row masking/gather).
O2. sample_hidden = mix(X2, mixer.hc_norm, mixer.down, mixer.up)    # [T,H]
    # mtp.hyper_connection_mixer, mix-only (no block_inject — none exists).
    # Its hc_norm is GROUPED per-stream (variance over H per stream) — the OPPOSITE
    # convention of pre_fc_norm_hidden in F4. Do not swap them.
O3. (no further norm — the mixer IS the final norm; there is no mtp.norm tensor)
O4. logits = sample_hidden @ lm_head.T          # the TRUNK's lm_head, [V]
```

Return `(logits, m_out)`.

### 1.6 Consumption audit (assert in code)

Per forward: F2..F6 consume the 4 head-input tensors once; A1–A3 consume the 4
attn_hc tensors + attention weights once; M1–M3 the 4 mlp_hc tensors + MoE weights
once; O2 the 3 mixer tensors once. Indexer tensors: zero uses (intentional).
`embed_tokens` once (I1), `lm_head` once (O4). Anything else touched ⇒ bug.

### 1.7 Source disagreements & resolutions

- **D1 Fusion form.** llama.cpp #27739: convert-time `eh_proj [2H,H]` concat-matmul.
  vLLM/TokenSpeed/SGLang: runtime add (`residual_linear_shared`). Mathematically
  identical per stream; checkpoint shapes (two `[2560,2560]`) mandate the ADD form
  (§1.2). Resolved: ADD, 4/4 sources agree on the math.
- **D2 embed/lm_head.** SGLang extraction mentions `mtp.embed_tokens` /
  `mtp.shared_head.head` (modules it *builds*); ground-truth inventory + vLLM
  confirm the checkpoint ships neither. Resolved: share the trunk's
  `embed_tokens` and `lm_head`.
- **D3 QSA in the head.** llama.cpp: dense, indexer loaded-but-unused ("tested,
  no throughput gain"). SGLang/vLLM: run QSA with step-0 index capture + reuse.
  Dense attention is a strict superset of QSA top-k selection (identical whenever
  context ≤ index_topk) and is the correct baseline for an implementation chasing
  correctness. Resolved: DENSE; QSA is a later optimization, never a correctness
  requirement.
- **D4 Positions.** llama.cpp + SGLang: the head decodes each input token at that
  token's absolute trunk position (rule in §2.2). vLLM's EAGLE plumbing uses
  target positions for the shifted ids (a uniform −1). A uniform shift is
  RoPE-relative-benign; MIXED conventions inside one implementation are not.
  Resolved: use the absolute-position rule of §2.2 everywhere.
- **D5 Naming.** "Qwen3.8-Flash-Next" is the public name; every codebase calls the
  arch `qwen4_exp` / Qwen4-Exp (LLM_TYPE_122B_A10B, 48 trunk layers).

---

## 2. The multi-step draft loop

### 2.1 What the trunk must expose

On EVERY target forward, capture per token the **pre-final-mixer** stream state:
the `[T, D]` tensor immediately BEFORE the trunk's model-level
`hyper_connection_mixer.mix(...)` collapses it to `[T, H]` (and before the trunk's
output-row gather/masking — you need rows for all verified tokens, not just the
sampled ones). Call row `i` of it `Hs_i`. This is what llama.cpp calls
`t_h_nextn` / "mtp_h_input", SGLang `hc_hidden_states`, vLLM `multi_hidden`
(`spec_hidden_size = hidden_size * hc_count = 10240`).

### 2.2 Pairing and position rule (the invariant)

Let positions `0..n-1` be target-decoded (each has an `Hs` row) and `t_next` be the
token the target just sampled (to be placed at position `n`).

> **Head input at position `p` is always `(token that sits at position p,
> Hs of position p−1)`, decoded at RoPE/KV position `p`.**
> The head's output at `p` is the candidate for position `p+1`.

Equivalently: the token id is always paired with the stream state of the token
*before* it ("shift the target embeddings right by one" — llama.cpp
speculative.cpp; "cat(target_ids[1:], next_token)" — vLLM/SGLang).

### 2.3 Head KV-cache policy (catch-up)

Keep `n_exact` = highest position whose head-KV cell was built from TARGET `Hs`
rows. After every verification round:

```
1. Truncate the head KV to positions ≤ n_exact.        # drop cells built from the
                                                       # head's own m_out feedback
2. Catch-up decode (one batched head forward, logits discarded):
   for each newly committed token at position p in (n_exact, n-1]:
       input (token_p, Hs_{p-1}) at position p
   First-ever round only: position 0 pairs with zeros (no Hs_{-1}); one cell,
   engines do the same, influence is negligible.
3. n_exact = n-1.
```

This mirrors llama.cpp's per-round catch-up decode and TokenSpeed's "rewrite
provisional tail KV with exact values". Never leave draft-feedback-built cells in
the cache across a verification boundary.

### 2.4 Draft steps

```
step 0:  (logits, M) = head(t_next,  Hs_{n-1}, pos = n)      ; d_1 = sample(logits)
step k:  (logits, M) = head(d_k,     M_prev,   pos = n + k)  ; d_{k+1} = sample(logits)
```

- The drafted token is fed BY ID and embedded inside the head via the trunk's
  `embed_tokens` (I1) — never feed a hidden state in place of the embedding.
- `M_prev` is the head's own `m_out` `[D]` from the previous step (O1) — the
  recurrence runs on the 4-stream state, not on `sample_hidden`.
- Position advances by exactly +1 per step; the head KV grows one cell per step.
- Sampling: greedy for acceptance debugging. Engine defaults: llama.cpp top-k 10
  with a p_min high-confidence cutoff; SGLang (steps=3, topk=1, draft=4); blog
  "MTP-213" = 2 steps / topk 1 / 3 draft tokens, accept length ≈ 3.3 incl. bonus.
- Stop when top prob < p_min or n_max drafts collected.

### 2.5 Verify and iterate

Target decodes `[t_next, d_1..d_m]` at positions `n..n+m` (emitting `Hs` rows for
all of them), accepts prefix length `a`, samples `t_next'` at position `n+a`.
Set `n ← n+a+1`, go to §2.3. Carry `(Hs_{n-1}, t_next')` across the boundary
(llama.cpp's `pending_h`).

---

## 3. Top 3 zero-acceptance bugs, in probability order

Zero (not merely low) acceptance means the head's argmax is essentially never the
target's token — the head is seeing garbage input or producing mis-scaled output.
Given proven machinery, the head-assembly candidates are:

### BUG 1 (most likely): wrong hidden feed — wrong capture point, tiled collapse, or transposed flatten
Generic spec-decode plumbing exposes the *collapsed* `[2560]` final hidden.
Failure variants, all shape-silent:
  (a) capturing the trunk hidden AFTER the final mixer (or after a final norm) and
      tiling it ×4 to fill 10240;
  (b) capturing before the mixer but flattening as `[H,S]` instead of `[S,H]`
      (element `s*H+h` vs `h*S+s`) anywhere across the trunk↔head boundary —
      including inside the head when reshaping for F5/A1/M1/O2;
  (c) capturing after output-row masking, so rows misalign with tokens.
Every variant feeds severely out-of-distribution input ⇒ ~0% acceptance.

### BUG 2: input fusion mis-assembled
Failure variants, all shape-silent:
  (a) `pre_fc_norm_hidden` applied per-stream (4 variances over 2560) instead of ONE
      variance over the full 10240 (F4) — the single easiest trap because every
      OTHER `[10240]` norm in the model (hc_norm, mixer norm) IS per-stream;
  (b) the projected embedding added to only one stream, or concatenated, instead of
      broadcast-added to all four (F6);
  (c) `fc_hidden` applied to a collapsed mean of the streams instead of per-stream
      (F5), or `h` normed with the trunk's replicate-embedding input path;
  (d) plain `w` instead of Gemma `(1+w)` in `pre_fc_norm_*` (only plausible if the
      head uses a different norm class than your proven trunk).

### BUG 3: final mixer / feedback tap mis-assembled
Failure variants:
  (a) replacing O2 with a plain RMSNorm + mean over streams (dropping the low-rank
      sigmoid gate), or borrowing the TRUNK's mixer weights instead of
      `mtp.hyper_connection_mixer.*`;
  (b) mixer `hc_norm` computed with ONE variance over 10240 (the F4 convention)
      instead of grouped per-stream — the exact inverse of BUG 2a;
  (c) SUM instead of MEAN over streams, or missing the ÷S before silu (mix step 3),
      breaking the trained scale of `sample_hidden` and hence the logits;
  (d) feeding the next draft step `sample_hidden` (2560, or tiled) instead of the
      pre-mixer `m_out` — this one uniquely leaves step-1 acceptance nonzero while
      zeroing steps ≥ 2, which is how you distinguish it.

Honorable mentions if all three pass: token/hidden pairing off by one (§2.2 —
pairing `(t_i, Hs_i)`); inject gate missing the ×2 or ÷S, or computed from raw
instead of normed streams; trunk rope theta reused instead of 1e7.

---

## 4. Cheap numeric sanity tests (one per bug)

Run all on real weights; each is < 1 minute and requires no training data beyond a
short prompt.

### T1 — mixer reconstruction identity (kills BUG 1, and validates mix() for BUG 3)
For the last token of any prompt, capture `v = Hs [10240]` exactly as you feed it to
the head. Apply **your own** `GatedResidual.mix` code with the **TARGET's**
model-level `hyper_connection_mixer` weights:

```
u = mix(v.reshape(1,4,2560), tgt_mixer.hc_norm, tgt_mixer.down, tgt_mixer.up)
assert max|u - h_final| <= 1e-3 * rms(h_final)
```

where `h_final [2560]` is the hidden your (working) trunk actually fed to `lm_head`
for that token. This passes ONLY when (capture point, stream layout, grouped norm,
÷S, silu/sigmoid order, gated MEAN) are all simultaneously right. Sub-probe for
tiling: `rows = v.reshape(4,2560)`; if all pairwise cosines > 0.999 you fed a tiled
collapsed hidden — real streams differ materially.

### T2 — fusion probes (kills BUG 2)
With real head weights, random `e [2560]`, random `X [4,2560]`:
  (a) **global-variance probe:** compute F4 on `X` and on `X` with stream 2 scaled
      ×10. ALL FOUR streams of the normed output must change (single 10240
      variance). If only stream 2 changes, you built the per-stream norm.
  (b) **broadcast/zero probes:** with `X = 0`: all four output streams of §1.2 must
      be IDENTICAL and equal `fc_embedding(GemmaRMSNorm(e, w_e))`. With `e = 0`:
      output stream `s` must equal `fc_hidden(F4(X)[s])` and streams must DIFFER.
  (c) **Gemma probe:** `pre_fc_norm_embedding.weight.mean()` and
      `pre_fc_norm_hidden.weight.mean()` — near 0 (|mean| ≲ 0.2) ⇒ `(1+w)`
      semantics required; near 1 ⇒ plain `w`. (Expected: near 0.)

### T3 — grouped-norm + feedback probes (kills BUG 3)
  (a) **grouped-variance probe** on `mtp.hyper_connection_mixer.hc_norm`: scale
      stream 2 of the input ×10; ONLY stream 2 of the normed tensor may change
      (per-stream variance) — the exact OPPOSITE expectation of T2a. If your code
      gives the same answer for T2a and T3a, one of the two norms is wrong.
  (b) **feedback probe:** at draft step 1, assert the fed-back vector has
      `dim == 10240` and `m.reshape(4,2560)` rows are NOT all pairwise-cosine
      > 0.999 (i.e. it is `m_out` from O1, not a tiled `sample_hidden`).
  (c) **step-profile probe:** measure per-step acceptance separately; nonzero at
      step 1 but zero at step ≥ 2 ⇒ BUG 3d specifically.

### T0 — end-to-end teacher forcing (run first; localizes everything)
Over a ~200-token prompt with trunk rows `Hs_i`: for each i, compute
`argmax(head(t_{i+1}, Hs_i, pos=i+1))` and score agreement with `t_{i+2}`
(no KV chaining needed — batch it with the catch-up path). Healthy head: ≳40–70%
top-1 agreement. ≲5% ⇒ forward broken (run T1→T2→T3 in order). If the SHIFTED
pairing `(t_i, Hs_i)` scores dramatically higher than the spec pairing, your
machinery's token/hidden pairing is off by one (§2.2), not the head.
