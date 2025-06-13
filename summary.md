Below is an expanded, implementation-focused summary of **Continuous Thought Machines (CTM)**, drawing on detailed architectural, algorithmic, and training specifics from the paper. Benchmarks and experimental results are only mentioned in passing; the emphasis is on how the model is built and operates under the hood.

---

## 1. Core Concept and Data Flow

1. **Internal Thought Dimension (Ticks)**

   * CTM introduces an internal recurrence over ticks $t = 1 \ldots T$, decoupled from any data timestep. This lets the model “think” multiple passes over a static input before producing outputs .

2. **High-Level Forward Pass**

   ```python
   # (Listing 1: Simplified CTM forward)
   kv = kv_proj(backbone(inputs))      # 1) Encode data → KV tokens
   pre_hist = pre_init.unsqueeze(0).repeat(B,1,1)  # B×D×M
   z = z_init.unsqueeze(0).repeat(B,1)            # B×D
   for t in range(T):
       # a) Compute action sync → query
       sync_a = compute_synch(post_hist, type="action")
       q = q_proj(sync_a)
       o = attn(q, kv, kv)                         # Cross-attention readout
       # b) Synapse MLP → new pre-activations
       pre = synapses(torch.cat([o, z], dim=-1))
       pre_hist = torch.cat([pre_hist[:,:,1:], pre.unsqueeze(-1)], dim=-1)
       # c) Neuron-Level Models → new post-activations
       z = neuron_level_models(pre_hist)
       post_hist.append(z)
       # d) Compute output sync → logits
       sync_o = compute_synch(post_hist, type="output")
       outputs.append(out_proj(sync_o))
   ```

   All components (backbone, projectors, attention, synapse MLP/U-Net, neuron models, output projector) are standard PyTorch modules .

---

## 2. Synapse Model (“Recurrent Weights”)

* **Purpose:** Share information across all $D$ neurons at each tick.
* **Structure:** A U-Net–style MLP of even depth $k$, bottlenecking down to width 16 then expanding back, with skip-connections and LayerNorm; dropout $p_{\text{drop}}=0.1$ (or task-specific) applied .
* **Computation:**

  $$
    a_t = f_{\theta}^{\text{syn}}\bigl([\;z_t\;;\;o_t\;]\bigr) \in \mathbb{R}^D
  $$

  where $z_t$ is the previous post-activation vector and $o_t$ the attention output .

---

## 3. Neuron-Level Models (NLMs)

* **Private MLP per Neuron:** Each of the $D$ neurons has its own MLP $g_{\theta}^d$ mapping its $M$-length pre-activation history to a scalar post-activation

  $$
    z_{t+1}^d = g_{\theta}^d\bigl(A_t^d\bigr),\quad A_t^d\in\mathbb{R}^M.
  $$
* **Implementation:** Two batched `einsum` calls compute all $D$ neurons in parallel: first to project history → hidden, then hidden → activation .
* **Parameter Cost:** Scales as $D\times(M\times H_{\text{dim}} + H_{\text{dim}})$, where $H_{\text{dim}}$ is the MLP hidden width (e.g.\ 32).

---

## 4. Neural Synchronization

1. **Full Matrix:** At tick $t$, stack post-activations into $Z_t\in\mathbb{R}^{D\times t}$; the raw sync matrix is $Z_t\,Z_t^\top$.
2. **Learnable Decay:** For each neuron-pair $(i,j)$, a learnable rate $r_{ij}\ge0$ rescales past contributions via

   $$
     \alpha^ {ij}_t = \sum_{\tau=1}^t e^{-r_{ij}(t-\tau)}\,z_{\tau i}\,z_{\tau j},\quad
     \beta^{ij}_t = \sum_{\tau=1}^t e^{-r_{ij}(t-\tau)},
   $$

   then $S_{t,ij} = \alpha^{ij}_t/\sqrt{\beta^{ij}_t}$ .
3. **Subsampling Strategies:**

   * **Dense pairing:** pick $J$ neurons, compute all $\tfrac{J(J+1)}2$ pairs.
   * **Semi-dense:** two subsets $J_1,J_2$, cross pair.
   * **Random pairing:** sample $D_{\text{out}}$ or $D_{\text{action}}$ pairs, allowing overlap and self-pairs to retain snapshot info .
4. **Efficiency:** Rather than $O(D^2 t)$ per tick, recursive updates yield $O(D_{\text{sub}})$ work per tick, where $D_{\text{sub}}\!\ll\!D$ .
5. **Usage:**

   * **Action sync** $\to$ query $q_t$ for attention
   * **Output sync** $\to$ logits $y_t$ via linear projections

---

## 5. Initialization & Continuous History

* **Learned Start States:**

  * $z_{\text{init}}\in\mathbb{R}^D$,
  * Pre-activation history $\in\mathbb{R}^{D\times M}$,
    both registered as `Parameter` and repeated across batch at each episode start .
* **Across Environment Steps (RL):** These traces are carried forward between environment steps so CTM can accumulate arbitrarily long histories .

---

## 6. Hyperparameters & Model Sizes

*(ImageNet-1K setup, as a representative example)*

* Backbone: ResNet-152 features (14×14 → K,V)
* $D=4096$, synapse depth $k=16$, $d_{\text{input}}=1024$, heads = 16
* $J_{\text{out}}=8196,\;J_{\text{action}}=2048,\;n_{\text{self}}=32$
* $T=50,\;M=25,\;d_{\text{hidden}}=64,\;p_{\text{drop}}=0.2$
* Total params: \~$200M$
* **Optimization:** Batch 64 across 8×H100, AdamW (5e-4 LR, 10k warmup, cosine decay), gradient clipping 20, no WD .

Variants for CIFAR-100, parity, RL and maze tasks adjust these (see Appendix D–G for full tables) .

---

## 7. Loss & Training

1. **Per-Tick Outputs:** CTM produces logits $y_t$ and normalized entropy “certainty” $C_t = 1 - (\mathrm{entropy}/\text{max\_entropy})$.
2. **Dual-Index Loss:** For each sample, let $t_1 = \arg\min_t L_t$ (min CE) and $t_2 = \arg\max_t C_t$. Final loss

   $$
     \mathcal{L} = \tfrac12\bigl(\,L_{t_1} + L_{t_2}\bigr)
   $$

   optionally reverting to final-tick-only if `use_most_certain=False`.
3. **Optimizer:** AdamW (or PPO for RL), linear warmup → cosine annealing, no explicit regularizer for adaptive compute .

---

## 8. Computational Considerations

* **Parameter Budget:** Despite per-neuron private models, total params remain comparable to baselines (e.g.\ LSTM) through careful matching of $D, M, H_{\text{dim}}$ and pairing sizes .
* **Run-Time Cost:**

  * Synapse: $O(D\,k)$ per tick
  * NLMs: $O(B\times D\times M \times H_{\text{dim}})$ via efficient `einsum`
  * Sync: $O(D_{\text{sub}})$ per tick via recursion
* **Memory:** Stores $M$-step pre-act history and $t$-step post-act history (pruned via FIFO or subsampled), plus learnable state traces.

---

This implementation unifies ideas from dynamic neural processes, per-neuron computation, and attention, enabling the CTM to “think” through internal recurrence, synchronize neuron dynamics, and adaptively decide when it’s certain enough to output—while remaining efficient and end-to-end trainable.
