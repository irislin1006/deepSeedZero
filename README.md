# deepSeedZero

This repository contains the environment setup used for reproducing the issue described in [Hugging Face Transformers Issue #32312](https://github.com/huggingface/transformers/issues/32312).

## Environment Information

To reproduce the issue, you will need the following environment specifications:

### My Environment Setup
- **GPU**: RTX 3090
- **Python Version**: 3.10.14
- **PyTorch Version**: 2.4.1+cu121
- **Transformers Version**: 4.45.0.dev0
- **DeepSpeed Version**: 0.9.3

### How to run?
After running setting up the environment, run
```bash
deepseed main.py
```
