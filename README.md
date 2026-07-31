# Transformer Reference Implementation: Algorithmic Reproduction of Attention Is All You Need

Author: Shriyans Pati

This repository provides PyTorch implementations and reproduction artifacts for the encoder-decoder Transformer architecture introduced in *Attention Is All You Need* by Vaswani et al. The code is intended for inspection, testing, and research-oriented reproduction of the core algorithmic components: scaled dot-product attention, multi-head attention, sinusoidal positional encodings, position-wise feed-forward networks, residual connections, layer normalization, autoregressive masking, and encoder-decoder cross-attention.

The `src/transformer_reference_implementation` package is a compact reference implementation. The `project`, `data`, `experiments`, `REPORT.md`, and `REPORT.pdf` files were added from the supplied archive and contain the broader reproduction study, logs, plots, and generated artifacts. Results in those artifacts should be interpreted only with the assumptions and limitations documented in `REPORT.md`.

## Scope

Implemented:

- Encoder-decoder Transformer modules in PyTorch.
- Sinusoidal positional encodings following the original formulation.
- Padding masks and causal masks for sequence-to-sequence modeling.
- A minimal greedy decoding utility.
- Unit tests for tensor shapes, masks, positional encodings, and decoding behavior.
- Imported reproduction code, experiment artifacts, and report files from the supplied project archive.

Not included:

- WMT14 preprocessing or tokenization recipes.
- Distributed training infrastructure.
- Checkpoints or pretrained weights.
- Empirical claims beyond the supplied artifacts and their documented constraints.

## Installation

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Unix-like shells, activate the environment with:

```bash
source .venv/bin/activate
```

## Quick Start

```python
import torch

from transformer_reference_implementation import TransformerConfig, TransformerModel

config = TransformerConfig(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1,
)

model = TransformerModel(config)

src = torch.randint(0, config.src_vocab_size, (2, 16))
tgt = torch.randint(0, config.tgt_vocab_size, (2, 12))

logits = model(src, tgt)
print(logits.shape)  # torch.Size([2, 12, 32000])
```

## Repository Layout

```text
transformer-reference-implementation/
├── .github/workflows/ci.yml
├── docs/
│   ├── architecture.md
│   ├── imported-readme.md
│   └── reproducibility.md
├── data/
├── experiments/
├── project/
├── src/transformer_reference_implementation/
│   ├── __init__.py
│   ├── config.py
│   ├── decoding.py
│   ├── masks.py
│   ├── modules.py
│   └── transformer.py
├── tests/
│   ├── test_decoding.py
│   ├── test_masks.py
│   └── test_transformer.py
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── REPORT.md
├── REPORT.pdf
└── README.md
```

## Imported Reproduction Artifacts

The supplied archive was added without replacing the curated repository README. Its original README is preserved at [docs/imported-readme.md](docs/imported-readme.md). The primary reproduction report is available as [REPORT.md](REPORT.md) and [REPORT.pdf](REPORT.pdf).

## Design Principles

The implementation prioritizes direct correspondence with the Transformer architecture over framework-specific abstraction. Modules are deliberately explicit so that tensor transformations, masking semantics, and residual pathways can be audited without relying on opaque wrappers.

The implementation uses pre-softmax additive masking with boolean masks. Mask values of `True` indicate positions that should be excluded from attention.

## Testing

Run the test suite with:

```bash
pytest
```

The tests verify implementation invariants rather than empirical model quality. Passing tests do not imply convergence, translation quality, or equivalence to a trained model from the original paper.

## Citation

If this repository is useful in academic work, cite it using the metadata in [CITATION.cff](CITATION.cff). The original Transformer paper should also be cited for the architecture:

```bibtex
@inproceedings{vaswani2017attention,
  title = {Attention Is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017}
}
```

## License

This project is released under the MIT License.
