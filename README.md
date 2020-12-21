# Attentive Adversarial Collaborative Filtering

## Environment Settings
- python==3.6
- tensorflow-gpu==1.14.0
- reckit==0.2.4
- numpy==1.19.1
- scipy==1.5.2
- pandas==1.0.5
- cython=0.29.21


## Dataset
We provide two processed datasets: Gowalla and CDs_and_Vinyl (CD).

## Quick Start

```bash
python run_model.py
```

If you run these code on a new dataset, please first compile DNSBPR code:
```bash
python setup.py build_ext --inplace
```