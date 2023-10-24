# Restricted_Boltzman_Machine

An implementation of a Generative Neural Network(GAN) called Restricted Boltzmann Machine (RBM) introduced in the following 2019 [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3384948). The goal is to do so with a minimum number of dependencies(currently only `numpy` and `scipy` as they go hand-in-hand anyway) and ideally using only optimized Linear Algebra operators(matrix multiplication, etc.).

## Current Status

Under development and testing before using the real-world market data. Ideally, the RBM should generate synthetic market data that can replicate the probability distribution of the original market data.