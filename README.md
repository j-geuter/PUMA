# PUMA

This is the code for PUMA (Progressive Unmasking) for the paper "Stop Training for the Worst: Progressive Unmasking Accelerates Masked Diffusion Training". PUMA gives a simple modification on the Masked Diffusion Models (MDMs) forward process and accelerates the pretraining.

## Quick Start

### 1. Install Environment

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate puma
```

### 2. Data preparation
We provide PUMA codebases for Sudoku and TinyGSM. 

- Sudoku: Download *sudoku-train-data.npy* and *sudoku-test-data.npy* and put them in the `data/sudoku_new` folder. 

- TinyGSM: Run `data/tiny_gsm.py` that gives you files `labels.bin`, `meta.json`, and `prompt_mask.bin` for pretraining in the desired `out_dir` directory.

### 3. Run Training
Submit a job using the SLURM script:

```bash
sbatch job.sh
```

The SLURM script calls `train.py`, which handles the training loop.

### 4. Configuration

Config files are located in `yaml_files/`. Edit these YAML files to adjust:
- Model architecture
- Training hyperparameters  
- Dataset settings
- Logging options
We provide one config each for PUMA and the baseline for the following three settings: Sudoku, TinyGSM (standard), TinyGSM (block diffusion).

### 5. Monitoring

Training logs and checkpoints are saved according to the paths specified in your config file. The training file also logs results to wandb.


## Explanation of each files
- `train.py`: unified file that handles the MDM pretraining (includes the vanilla MDM pretraining). Self-includes the evaluation accuracy logging.
- `sampling.py`: sampling for a given MDM
- `progressive.py`': PUMA via batch streaming implementation. The implementation detail can be found in Section 3.2 and Appendix B.1.
- `progressive_block.py`: PUMA implementation for block diffusion. 
- `model`: our Qwen2-style attention implementations
- `eval`: eval util functions for Sudoku / TinyGSM