# Attention-GRU CSI-HAR Experiments

## Overview  
This Jupyter notebook implements an attention-augmented Gated Recurrent Unit (GRU) model for Human Activity Recognition (HAR) using Channel State Information (CSI) collected from a Raspberry Pi 4. It performs leave-one-user-out cross-validation over seven activities (`walk`, `run`, `fall`, `lie down`, `sitdown`, `standup`, `bend`), compares multiple experiments (baseline GRU, Attention-GRU, CNN+GRU, etc.), and reports metrics and plots.

---

## Features  
- **Data Loader**: per-file normalization + sliding-window segmentation  
- **Model Variants**:  
  - Baseline GRU  
  - Attention-augmented GRU  
  - CNN + GRU hybrid  
- **Cross-Validation**: leave-one-user-out (3 users)  
- **Learning-Rate Finder**: automated LR sweep to pick optimal LR  
- **Augmentation**: time-shift & noise, optional MixUp  
- **Metrics & Plots**: accuracy curves, loss curves, classification reports, confusion matrices  

---

## Requirements  
- Python 3.10+  
- PyTorch 1.13+  
- NumPy, pandas, matplotlib  
- scikit-learn  
- tqdm (optional)  

You can create a conda environment:

```bash
conda create -n csi-har python=3.10 numpy pandas matplotlib scikit-learn pytorch -c pytorch
conda activate csi-har
pip install tqdm


CSI-HAR-Dataset/
├─ walk/
│   ├─ user_1_sample_01_walk.csv
│   ├─ …
├─ run/
│   ├─ user_1_sample_01_run.csv
│   ├─ …
├─ fall/
│   ├─ …
└─ … (sitdown, standup, bend, lie down)


DATASET_ROOT = r'F:/Datasets/CSI-HAR-Dataset-Raspberry-Pi-4/CSI-HAR-Dataset'
