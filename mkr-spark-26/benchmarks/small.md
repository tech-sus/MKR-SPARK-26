# MKR - SPARK 26 benchmark (preset 'small')
threads: 1   os: linux   arch: x86_64
matmul 256x256x256: 19.75 GFLOP/s
training step (fwd+bwd, batch 1 x ctx 64): 596 ms median, 107 tokens/s; peak tensor memory 181.9 MB
inference f32: 243 tokens/s, weights 76.4 MB, state 1.6 MB
inference int8: 264 tokens/s, weights 19.6 MB, state 1.6 MB
parameters: 19.09M   peak process RAM: 237.4 MB
