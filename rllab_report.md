# RLLAB
### Normalization
  - actions -> [-1,1] (if box)
  - obs (optional): (x-μ) / σ
  - reward (optional): x / σ
  -  both use running norm
  - advantage:
    - center_adv -> normalize for current batch (?)
    - positive_adv -> -np.min (??)

### Advantage
  - estimated using the old value function (yay!)

### Loss
  - uses "adaptive KL" penalty (L^KLPEN)

### Baseline
### RNN policy


# OpenAI Baselines
### Normalization
  - obs: clip( (x-μ) / σ , -5 , 5 )
    - with running norm
  - reward: none
  - advantage: (x-μ) / σ
    - calculated at each iteration
### Advantage
  - estimated using the old value function (yay!)
  - does not correct the advantage for the last step in episode
### batches
  - iterates over the whole data n (~10) times to update the optim with batches of size k (~64)
### Loss
  - uses "clipped surrogate" objective (L^CLIP)
  - standardized advantage function estimate (??)
### Optim
  - uses Adam (+MPI)
  - anneals clipping ϵ with lr (??)
### Network size
  - same size for value function and policy


# Questions
  - why standardize the advantages?
  - why annealing clipping ϵ