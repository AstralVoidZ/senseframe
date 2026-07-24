"""L3 算法行为测试。

锚点来源：论文 / 官方实现。
- MAE: He et al., 2022《Masked Autoencoders Are Scalable Vision Learners》
- DANN: Ganin & Lempitsky, 2015《Unsupervised Domain Adaptation by Backpropagation》
- DARTS: Liu et al., 2019《DARTS: Differentiable Architecture Search》
- ASHA: Li et al., 2018《Massively Parallel Hyperparameter Tuning》
- AutoAugment: Cubuk et al., 2019

禁止：源码自身常量作为断言目标（如 model._mask_ratio == 0.75 不引用论文）。
"""
