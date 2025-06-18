# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import lru_cache

import numpy as np
from unicore.data import BaseWrapperDataset

from . import data_utils


class AffinityPocketDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket_atoms,
        pocket_coordinates,
        is_train=False,
        pocket="pocket"
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.is_train = is_train
        self.pocket=pocket
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        pocket_atoms = np.array(
            [self.pocket_atom(item) for item in self.dataset[index][self.pocket_atoms]]
        )
        ori_length = len(pocket_atoms)
        pocket_coordinates = np.stack(self.dataset[index][self.pocket_coordinates])
        if self.pocket in self.dataset[index]:
            pocket = self.dataset[index][self.pocket]
        else:
            pocket = ""
        return {
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_pocket_coordinates": pocket_coordinates.astype(np.float32),#placeholder
            "pocket": pocket,
            "ori_length": ori_length
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class PeptideAffinityDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        pocket1_atoms,
        pocket1_coordinates,
        pocket2_atoms,
        pocket2_coordinates,
        affinity,
        is_train=False,
        pocket1="pocket1",
        pocket2="pocket2"
    ):
        self.dataset = dataset
        self.seed = seed
        self.pocket1_atoms = pocket1_atoms
        self.pocket1_coordinates = pocket1_coordinates
        self.pocket2_atoms = pocket2_atoms
        self.pocket2_coordinates = pocket2_coordinates
        self.affinity = affinity
        self.is_train = is_train
        self.pocket1 = pocket1
        self.pocket2 = pocket2
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def pocket_atom(self, atom):
        """处理原子标识符，去除数字前缀"""
        if atom[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            return atom[1]
        else:
            return atom[0]

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):

        data_item = self.dataset[index]
        
        # 继续原有的处理逻辑
        pocket1_atoms_raw = data_item[self.pocket1_atoms]
        if isinstance(pocket1_atoms_raw, int):
            raise TypeError(
                f"pocket1_atoms 应为字符串列表，实际为 int: {pocket1_atoms_raw}，请检查上游数据处理流程。"
            )
        pocket1_atoms = np.array(
            [self.pocket_atom(item) for item in pocket1_atoms_raw]
        )
        ori_pocket1_length = len(pocket1_atoms)
        
        # 处理第一个口袋的坐标
        pocket1_coords_data = data_item[self.pocket1_coordinates]
        if isinstance(pocket1_coords_data, list) and len(pocket1_coords_data) > 0:
            # 如果有多个构象，随机选择一个
            size1 = len(pocket1_coords_data)
            if self.is_train:
                with data_utils.numpy_seed(self.seed, epoch, index):
                    sample_idx1 = np.random.randint(size1)
            else:
                with data_utils.numpy_seed(self.seed, 1, index):
                    sample_idx1 = np.random.randint(size1)
            pocket1_coordinates = pocket1_coords_data[sample_idx1]
        else:
            pocket1_coordinates = pocket1_coords_data
        
        if isinstance(pocket1_coordinates, list):
            pocket1_coordinates = np.stack(pocket1_coordinates)
        else:
            pocket1_coordinates = np.array(pocket1_coordinates)

        # 处理第二个口袋
        pocket2_atoms = np.array(
            [self.pocket_atom(item) for item in data_item[self.pocket2_atoms]]
        )
        ori_pocket2_length = len(pocket2_atoms)
        
        # 处理第二个口袋的坐标
        pocket2_coords_data = data_item[self.pocket2_coordinates]
        if isinstance(pocket2_coords_data, list) and len(pocket2_coords_data) > 0:
            # 如果有多个构象，随机选择一个
            size2 = len(pocket2_coords_data)
            if self.is_train:
                with data_utils.numpy_seed(self.seed, epoch, index + 1000):  # 不同的seed避免相关性
                    sample_idx2 = np.random.randint(size2)
            else:
                with data_utils.numpy_seed(self.seed, 1, index + 1000):
                    sample_idx2 = np.random.randint(size2)
            pocket2_coordinates = pocket2_coords_data[sample_idx2]
        else:
            pocket2_coordinates = pocket2_coords_data
            
        if isinstance(pocket2_coordinates, list):
            pocket2_coordinates = np.stack(pocket2_coordinates)
        else:
            pocket2_coordinates = np.array(pocket2_coordinates)

        # 获取口袋标识符
        pocket1_id = data_item.get(self.pocket1, "")
        pocket2_id = data_item.get(self.pocket2, "")
        
        # 获取相似性标签
        if self.affinity in data_item:
            affinity = float(data_item[self.affinity])
        else:
            affinity = 1  # 默认为正样本

        return {
            # 第一个口袋信息
            "pocket1_atoms": pocket1_atoms,
            "pocket1_coordinates": pocket1_coordinates.astype(np.float32),
            "pocket1": pocket1_id,
            "ori_pocket1_length": ori_pocket1_length,
            
            # 第二个口袋信息
            "pocket2_atoms": pocket2_atoms,
            "pocket2_coordinates": pocket2_coordinates.astype(np.float32),
            "pocket2": pocket2_id,
            "ori_pocket2_length": ori_pocket2_length,
            
            # 相似性标签
            "affinity": affinity,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)