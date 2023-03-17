This repo is mainly based on the original [CORGI-PM repo](https://github.com/yizhilll/CORGI-PM). 

---

# Introduction

**CORGI-PM**🐶 is a Chinese cOrpus foR Gender bIas Probing and Mitigation, which contains **32.9k** sentences with high-quality labels derived by following an annotation scheme specifically developed for gender bias in the Chinese context.

We address three challenges for automatic textual gender bias mitigation, which requires the models to detect, classify, and mitigate textual gender bias.

# Experiments

## Bias Detection

We formulate the bias detection tasks as binary classification. To run the codes:

```shell
python -u src/run_classification.py detection 
```

## Bias Classification

Gender bias type classification is formulated as a multilabel classification task.

```
python -u src/run_classification.py multilabel  
```

# Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2301.00395,
  doi = {10.48550/ARXIV.2301.00395},
  url = {https://arxiv.org/abs/2301.00395},
  author = {Zhang, Ge and Li, Yizhi and Wu, Yaoyao and Zhang, Linyuan and Lin, Chenghua and Geng, Jiayi and Wang, Shi and Fu, Jie},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Computers and Society (cs.CY), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {CORGI-PM: A Chinese Corpus For Gender Bias Probing and Mitigation},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
Notes:

I will (try to) sort out the full version of codes after the release of the final grade.
