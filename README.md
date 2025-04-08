<h1 align="center">
  DRK Splatting

</h1>



<div align="center">




## Environment Setup

### Create and Activate Python Environment
#### Using Conda:
```bash
conda create -n drkenv python=3.9  # (Python >= 3.8)
conda activate drkenv
```

#### Using Virtualenv:
```bash
virtualenv drkenv -p python3.9  # (Python >= 3.8)
source drkenv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
cd submodules/depth-diff-gaussian-rasterization
python setup.py install && pip install .
cd ../drk_splatting
python setup.py install && pip install .
cd ../simple-knn
python setup.py install && pip install .
cd ../..
```





## Running the Code

### Commands
Run the following commands in your terminal:

#### Training:
```bash
CUDA_VISIBLE_DEVICES=${GPU} python train.py -s ${PATH_TO_DATA} -m ${LOG_PATH} --eval --gs_type DRK --kernel_density dense --cache_sort  # Optional: --gui --is_unbounded
```

#### Evaluation:
```bash
CUDA_VISIBLE_DEVICES=${GPU} python train.py -s ${PATH_TO_DATA} -m ${LOG_PATH} --eval --gs_type DRK --kernel_density dense --cache_sort --metric
```

### Command Options:
- `--kernel_density`: Specifies the primitive density (number) for reconstruction. Choose from `dense`, `middle`, or `sparse`.
- `--cache_sort`: (Optional) Use cache sorting to avoid popping artifacts and slightly increase PSNR (approx. +0.1dB). Ensure consistency between training and evaluation. Note: In specular scenes, disabling cache-sort may yield better results as highlights are better modeled without strict sorting.
- `--is_unbounded`: Use different hyperparameters for unbounded scenes (e.g., Mip360).
- `--gui`: Enables an interactive visualization UI. Toggle cache-sorting, tile-culling, and view different rendering modes (normal, depth, alpha) via the control panel.

### Batch Scripts
Scripts for evaluating all scenes in the dataset are provided in the [scripts](./scripts) folder. Modify the paths in the scripts before running them.

```bash
python ./scripts/diverse_script.py  # For DiverseScenes
python ./scripts/mip360_script.py   # For MipNeRF-360
```

---


```
