# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses semantic versioning.

## [0.1.0] - 2026-07-30

### Added

- Initial PyTorch reference implementation of the encoder-decoder Transformer.
- Sinusoidal positional encoding, multi-head attention, feed-forward blocks, encoder layers, decoder layers, and full sequence-to-sequence model.
- Boolean padding and causal mask utilities.
- Greedy autoregressive decoding helper.
- Unit tests for masks, model shapes, attention weights, and decoding termination.
- Documentation for architecture and reproducibility boundaries.
