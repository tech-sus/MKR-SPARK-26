# MKR - SPARK 26 benchmark (preset 'nano')
threads: 1   os: linux   arch: x86_64
matmul 256x256x256: 14.49 GFLOP/s
training step (fwd+bwd, batch 4 x ctx 128): 41 ms median, 12453 tokens/s; peak tensor memory 12.0 MB
inference f32: 32549 tokens/s, weights 670 KB, state 17 KB
inference int8: 26466 tokens/s, weights 228 KB, state 17 KB
parameters: 0.17M   peak process RAM: 20.2 MB
