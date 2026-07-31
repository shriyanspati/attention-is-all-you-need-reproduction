# Reproducibility Notes

This repository is an algorithmic reference implementation. It is not a full reproduction package for the empirical results in *Attention Is All You Need*.

## What Is Reproducible Here

The following properties can be reproduced from the repository alone:

- Tensor-level behavior of the implemented Transformer modules.
- Mask construction for padding and autoregressive decoding.
- Forward-pass shape contracts.
- Greedy decoding control flow.
- Unit-test coverage for implementation invariants.

## What Is Not Claimed

This repository does not claim:

- WMT14 English-German or English-French BLEU scores.
- Training compute equivalence to the original paper.
- Hyperparameter equivalence beyond the configurable architecture parameters.
- Numerical equivalence to any released checkpoint.
- State-of-the-art performance on any benchmark.

## Missing Experimental Artifacts

The current repository does not include dataset manifests, preprocessing scripts, trained checkpoints, random seeds for reported experiments, or evaluation logs. Those artifacts are required before empirical results can be reported responsibly.

## Adding Results

Any future result should include:

- Dataset name, version, and preprocessing details.
- Tokenizer training procedure and vocabulary size.
- Complete training configuration.
- Hardware and software environment.
- Random seeds.
- Checkpoint selection rule.
- Evaluation script and metric implementation.
- Raw logs or immutable artifacts sufficient to audit the claim.
