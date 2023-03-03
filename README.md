# Attentive Adversarial Collaborative Filtering

This is our Tensorflow implementation for our AACF paper:
> Zhongchuan Sun, Bin Wu, Shizhe Hu, Mingming, Zhang, and Yangdong, Ye. Attentive Adversarial Collaborative Filtering, IEEE Transactions on Systems, Man, and Cybernetics: Systems, Accept.

## Environment Requirements

- python==3.6
- tensorflow-gpu==1.14.0
- reckit==0.2.4
- numpy==1.19.1
- scipy==1.5.2
- pandas==1.0.5
- cython=0.29.21

## Quick Start

```bash
python run_model.py
```

If you run these code on a new dataset, please first compile DNSBPR code:

```bash
python setup.py build_ext --inplace
```

## Citation

If you find this useful for your research, please kindly cite the following paper.

```bibtex
@article{tsmc:2023:aacf,
  title   = {Attentive Adversarial Collaborative Filtering},
  author  = {Sun, Zhongchuan and Wu, Bin and Hu, Shizhe and Zhang, Mingming and Ye, Yangdong},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume  = {},
  pages   = {1--13},
  year    = {2023},
  doi     = {10.1109/TSMC.2023.3241083}
}
```
