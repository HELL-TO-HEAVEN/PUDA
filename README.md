# PUDA

This is a demenstrative implementation of our IJCAI 2022 [paper](https://arxiv.org/abs/2205.00904) Positive-Unlabeled Learning with Adversarial Data Augmentation for Knowledge Graph Completion (PUDA).

## Abstract

Most real-world knowledge graphs (KG) are far from complete and comprehensive. This problem has motivated efforts in predicting the most plausible missing facts to complete a given KG, i.e., knowledge graph completion (KGC). However, existing KGC methods suffer from two main issues, 1) $\textit{the false negative issue}$, i.e., the  sampled negative training instances may include potential true facts; and 2) the $\textit{data sparsity issue}$, i.e., true facts account for only a tiny part of all possible facts. To this end, we propose $\underline{p}\text{ositive-}\underline{u}\text{nlabeled learning with adversarial } \underline{d}\text{ata } \underline{a}\text{ugmentation}$ (PUDA) for KGC. In particular, PUDA tailors positive-unlabeled risk estimator for the KGC task to deal with the false negative issue. Furthermore, to address the data sparsity issue, PUDA achieves a data augmentation strategy by unifying adversarial training and positive-unlabeled learning under the positive-unlabeled minimax game. Extensive experimental results on real-world benchmark datasets demonstrate the effectiveness and compatibility of our proposed method. 

## Requirments
* python == 3.8.5
* torch == 1.8.1
* numpy == 1.19.2
* pandas == 1.0.1
* tqdm == 4.61.0
  
## Run

    nohup python ./run.py --data_root your_data_root >your_log_file 2>&1 &

## Reference

```
@article{tang2022positive,
  title={Positive-Unlabeled Learning with Adversarial Data Augmentation for Knowledge Graph Completion},
  author={Tang, Zhenwei and Pei, Shichao and Zhang, Zhao and Zhu, Yongchun and Zhuang, Fuzhen and Hoehndorf, Robert and Zhang, Xiangliang},
  journal={arXiv preprint arXiv:2205.00904},
  year={2022}
}
```
