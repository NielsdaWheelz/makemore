# makemore

learning project following [andrej karpathy's zero to hero](https://www.youtube.com/watch?v=TCH_1BHY58I) series. builds character-level language models that generate plausible names.

## what it does

implements neural networks from scratch to predict the next character in a name. starts simple (bigram) and progressively adds complexity (mlp, backprop, batch normalization). trains on `names.txt` dataset.

variants (in order of complexity):
- `01-bigram.py` - bigram model with neural net formulation, manual gradient descent
- `02-bigram-clean.py` - cleaner bigram with one-hot encoding, 100 training steps
- `03-mlp.py` - multi-layer perceptron with embeddings, 10k steps
- `04-mlp-hyperparams.py` - mlp with train/dev/test splits, hyperparameter experiments, 200k steps
- `05-mlp-variant.py` - alternative mlp architecture (10-dim embeddings, 200-unit hidden layer)

## getting started

### first time setup (any python project)

1. **check python is installed**
   ```bash
   python3 --version
   ```
   if not installed, get it from python.org or use homebrew: `brew install python3`

2. **install uv** (fast package manager, replaces pip)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   or: `brew install uv`

3. **create virtual environment** (isolates project dependencies)
   ```bash
   uv venv
   ```
   creates `.venv/` directory. this keeps packages separate from system python.

4. **activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```
   you'll see `(.venv)` in your terminal prompt. run this every time you open a new terminal session.

5. **install dependencies**

   if `requirements.txt` exists:
   ```bash
   uv pip install -r requirements.txt
   ```

   or install packages directly:
   ```bash
   uv pip install package1 package2
   ```

6. **deactivate when done**
   ```bash
   deactivate
   ```

### running this project

```bash
source .venv/bin/activate
python makemore/01-bigram.py
```

start with `01-bigram.py` and work through in order. the later files (04, 05) train for 200k steps and take a few minutes.

## dependencies

- torch - neural network framework
- matplotlib - plotting losses
- rich - pretty printing
- names.txt - training data (included)

## common gotchas

- forgot to activate venv? you'll get "module not found" errors
- `.venv/` should be in `.gitignore` (don't commit it)
- `requirements.txt` should be committed (do commit it)
- if you move the project, delete `.venv/` and recreate it
