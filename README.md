# Bengali-ASR

**Bengali-ASR** provides scripts and notebooks for building an automatic speech recognition (ASR) system for the Bengali language.  The project uses the Whisper architecture with JAX/TPU as well as PyTorch implementations based on Wav2Vec2.

## Project Structure
- `setup.sh` &ndash; installs Python 3.11 and all required packages (JAX, Transformers, PyTorch, etc.).
- `download.sh` &ndash; downloads training data from Kaggle and other public sources.
- `functions.py`, `functions_infer.py` &ndash; dataset utilities and dataloaders.
- `run_train.py` &ndash; main JAX training script for the Whisper model.
- `run_train_txt.py` &ndash; optional text-only training of the decoder.
- `model_wav2vec_CTC.py` &ndash; PyTorch approach using Wav2Vec2 with a CTC head.
- `*.ipynb` &ndash; Jupyter notebooks with experiments and evaluations.

## Setup

1. **Install dependencies**

   ```bash
   bash setup.sh
   ```

2. **Download data**

   ```bash
   bash download.sh
   ```
   Place your `kaggle.json` credentials in the project root before running the script.

3. **Activate the virtual environment**

   ```bash
   source ~/.venv311/bin/activate
   ```

## Training

### Whisper (JAX/TPU)

Edit hyperparameters in `run_train.py` as necessary, then execute:

```bash
python run_train.py
```

### Text-only fine-tuning

For additional training using only text data, run:

```bash
python run_train_txt.py
```

### Wav2Vec2 CTC (PyTorch)

The file `model_wav2vec_CTC.py` contains a PyTorch implementation with a CTC loss.  Run it directly after adjusting paths:

```bash
python model_wav2vec_CTC.py
```

## Inference

`functions_infer.py` shows how to create an inference dataset and collate function.  See the notebooks for end-to-end examples.

## Notebooks

Open the Jupyter notebooks for exploration and evaluation:

```bash
jupyter lab
```

## License

This repository is provided for research purposes. Please review the licenses of the datasets used before redistribution.
