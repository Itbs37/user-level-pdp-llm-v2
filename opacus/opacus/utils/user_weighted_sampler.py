"""
User-level Poisson sampler with personalized per-user sampling rates.

This module implements a sampler where each user is independently
sampled with their own probability, generalizing standard Poisson
subsampling to support personalized differential privacy at the
user level.

Based on:
    Boenisch, F., Mühl, C., Dziedzic, A., Rinberg, R., & Papernot, N.
    "Have it your way: Individualized Privacy Assignment for DP-SGD."
    NeurIPS 2023.
    https://github.com/sprintml/idp-sgd
    Original file: opacus/opacus/utils/weighted_sampler.py
    Licensed under Apache License 2.0.

Modifications from the original:
    - Sample-level (per-record) -> User-level (per-user) semantics
    - Renamed variables and class to reflect user-level usage
"""

from typing import List
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import Sampler

class UserLevelPoissonSampler(Sampler[List[int]]):
    """各ユーザに異なる sampling rate が割り当てられていて、それに基づいてサンプリングを行う。
    
    用途:
    提案手法の user 単位の sampling。
    各ユーザーが異なる ε を持つ場合、それに応じた sample rate を渡して使う。

    入力:
    pu_sample_rates (ndaaray): 各ユーザのサンプリング確率。
    generator: 乱数生成器

    出力:
    各バッチで、ポアソンサンプリンで選ばれたユーザーのインデックスを返す。
    """

    def __init__(self, *, pu_sample_rates: ndarray, generator=None):
        """
        Args:
            pu_sample_rates: 各 user の sampling 確率 (numpy array)
            generator: 乱数生成器
        """
        self.pu_sample_rates = pu_sample_rates
        self.generator = generator
        self.unique_sample_rates = np.unique(self.pu_sample_rates)
        self.num_users = len(self.pu_sample_rates)
        assert all(self.pu_sample_rates >= 0) and all(
            self.pu_sample_rates < 1
        ), "pu_sample_rates must be >=0 and <1!"
    
    def __len__(self):
        """1 epoch あたりの batch 数（期待値ベース）を返す。
        
        計算: 1 / (各群比率と sample rate の加重平均)
        """
        ratios = np.array([sum(self.pu_sample_rates == rate) / self.num_users
                           for rate in self.unique_sample_rates])
        return round(1 / np.dot(ratios, self.unique_sample_rates))

    def _iter__(self):
        """1 epoch分のバッチを順次yieldする。
        各バッチで、全ユーザに独立にポアソンサンプリング。
        選ばれたユーザーのインデックスをリストにして1バッチとしてyieldする。
        """
        num_batches = len(self)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_users, generator=self.generator)
                < torch.Tensor(self.pu_sample_rates)
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices
            num_batches -= 1