# MKR - SPARK 26 benchmark (preset 'micro')
threads: 1   os: linux   arch: x86_64
matmul 256x256x256: 14.85 GFLOP/s
training step (fwd+bwd, batch 4 x ctx 128): 309 ms median, 1657 tokens/s; peak tensor memory 48.8 MB
inference f32: 4064 tokens/s, weights 4.5 MB, state 133 KB
inference int8: 4690 tokens/s, weights 1.3 MB, state 133 KB
parameters: 1.12M   peak process RAM: 61.2 MB
