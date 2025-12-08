<!-- Copilot instructions for NLP_TRANSFORMER_PROJECT -->

# Copilot guidance — NLP_TRANSFORMER_PROJECT

Goal: Help AI coding agents be immediately productive editing, testing and extending this PyTorch Transformer prototype.

- **Run / test quickly**: this repo treats `src/` as the package root. To run the provided smoke test do either:

  - Change directory into `src/` then run the test script:
    ```powershell
    cd src
    python test_model.py
    ```
  - Or from project root set `PYTHONPATH` to `src` so top-level imports like `model.transformer` resolve:
    ```powershell
    $env:PYTHONPATH = 'src'; python -m test_model
    ```

- **Dependencies**: see `requirements.txt` (primary packages: `torch`, `numpy`, `datasets`). Use a virtualenv or conda; tests expect GPU if available (config auto-detects device in `configs/config.py`).

- **Big-picture architecture**:

  - `src/` is the Python package root. The core model lives under `src/model/` with these responsibilities:
    - `transformer.py`: orchestration, masks, and weight init (entry point for the model API).
    - `encoder.py`, `decoder.py`: stacked layer implementations (embedding, positional encoding, layer stacks).
    - `attention.py`: `MultiHeadAttention` implementation used by both encoder/decoder.
    - `positional_encoding.py`: sinusoidal PE module.
  - `configs/config.py` holds experiment hyperparameters (device detection, sizes). `src/test_model.py` is a light smoke test that instantiates `Transformer` and runs a forward pass.

- **Project-specific conventions / gotchas** (important to follow):

  - Import style is inconsistent across files (some modules use relative imports like `from .encoder import Encoder`, others use bare names like `from attention import MultiHeadAttention`). To avoid import errors, run tests from `src/` or set `PYTHONPATH=src` as shown above.
  - The `Transformer` constructor expects padding token indices (`src_pad_idx`, `trg_pad_idx`) and returns logits with shape `[batch, trg_len, trg_vocab_size]`. Many tests depend on these shapes.
  - Mask shapes: `make_src_mask` returns `[batch, 1, 1, src_len]`; `make_trg_mask` composes padding mask & look-ahead and returns shape compatible with attention modules. When editing masks, match these shapes.
  - Xavier init is applied in `Transformer._init_weights()` — keep this if you change parameter initialization.

- **When changing or adding modules**:

  - Prefer package-relative imports (e.g. `from .attention import MultiHeadAttention` inside `src/model/`) so modules work both as package and when `src` is on `PYTHONPATH`.
  - Unit-test small modules in place (see simple test blocks in `attention.py` and `positional_encoding.py`). For new unit tests, follow the pattern in `src/test_model.py` (small, deterministic dummy tensors and shape assertions).

- **Edit examples** (concrete snippets the agent can use):

  - Add a debug assertion in `transformer.py` forward to check mask shapes:
    ```py
    assert src_mask.dim() == 4 and trg_mask.dim() == 3 or trg_mask.dim() == 4
    ```
  - When adding a new argument to `Encoder.__init__`, update `Transformer.__init__` to forward parameters to keep API consistent.

- **Where to look first when debugging**:
  - Shape/size mismatch: `src/model/encoder.py` and `src/model/decoder.py` (embedding scaling and positional encoding steps).
  - Masking issues / runtime errors during forward: `src/model/transformer.py` and `src/model/attention.py` (mask application uses `masked_fill(mask == 0, -1e9)`).

If anything here is unclear or you want more details (testing commands, preferred import style, CI hooks), tell me which area to expand and I will iterate.
