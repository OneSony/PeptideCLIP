# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from IPython import embed as debug_embedded
import logging
import os
from collections.abc import Iterable
from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (PeptideAffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityPocketDataset, ResamplingDataset)
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio*n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp>= num:
                break
    return (tp*n)/(p*fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    return res2

def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score and other metrics for pocket similarity evaluation.
    """
    scores = np.expand_dims(y_score, axis=1)
    y_true_array = np.array(y_true)
    y_true_expanded = np.expand_dims(y_true_array, axis=1)
    scores = np.concatenate((scores, y_true_expanded), axis=1)
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    
    # 计算各个比例下的positive样本统计
    total_samples = len(y_true)
    total_positives = int(np.sum(y_true_array))
    
    index = np.argsort(y_score)[::-1]
    auc = CalcAUC(scores, 1)
    
    ratios = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ef = {}
    positive_counts = {}
    
    for i, ratio in enumerate(ratios):
        # 计算在这个比例下选择的样本数和其中positive的数量
        num_selected = int(total_samples * ratio)
        if num_selected == 0:
            num_selected = 1  # 至少选择1个样本
        
        selected_indices = index[:num_selected]
        positives_found = np.sum(y_true_array[selected_indices])
        
        # 手动计算EF值以确保一致性
        expected_positives = total_positives * ratio
        if expected_positives > 0:
            ef_calculated = positives_found / expected_positives
        else:
            ef_calculated = 0.0
        
        # 使用我们自己计算的EF值
        ef[str(ratio)] = ef_calculated
        positive_counts[str(ratio)] = int(positives_found)
        positive_counts[f"{ratio}_total"] = num_selected
    
    # 添加总体统计信息
    positive_counts["total_positives"] = total_positives
    positive_counts["total_samples"] = total_samples
    
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    return auc, bedroc, ef, re_list, positive_counts


@register_task("peptideclip")
class PeptideCLIP(UnicoreTask):
    """Task for training pocket-pocket contrastive learning models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-pocket1-model",
            default=None,
            type=str,
            help="pretrained pocket model path for encoder 1",
        )
        parser.add_argument(
            "--finetune-pocket2-model",
            default=None,
            type=str,
            help="pretrained pocket model path for encoder 2",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between pockets",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--dynamic-batch-crop",
            action="store_true",
            help="dynamically crop pockets in batch to same size for efficiency",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument(
            "--batch-log-dir",
            default=None,
            type=str,
            help="Directory to save batch pocket length logs (optional)"
        )

    def __init__(self, args, pocket_dictionary):
        super().__init__(args)
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.batch_log_dir = getattr(args, 'batch_log_dir', None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load pocket-pocket paired dataset.
        Expected data format:
        'pocket1', 'pocket1_atoms', 'pocket1_coordinates', 
        'pocket2', 'pocket2_atoms', 'pocket2_coordinates', 'label'
        """
        data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        
        if split.startswith("train"):
            pocket1_dataset = KeyDataset(dataset, "pocket1")
            pocket2_dataset = KeyDataset(dataset, "pocket2")
            
            dataset = PeptideAffinityDataset(
                dataset,
                self.args.seed,
                "pocket1_atoms",
                "pocket1_coordinates",
                "pocket2_atoms",
                "pocket2_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            # 是对的, 在PeptideAffinityDataset中把label改成了affinity
            
        else:
            dataset = PeptideAffinityDataset(
                dataset,
                self.args.seed,
                "pocket1_atoms",
                "pocket1_coordinates",
                "pocket2_atoms",
                "pocket2_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            pocket1_dataset = KeyDataset(dataset, "pocket1")
            pocket2_dataset = KeyDataset(dataset, "pocket2")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # 处理第一个口袋
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket1_atoms",
            "pocket1_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket1_atoms",
            "pocket1_coordinates",
            self.args.max_pocket_atoms,
        )

        # 处理第二个口袋
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket2_atoms",
            "pocket2_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket2_atoms",
            "pocket2_coordinates",
            self.args.max_pocket_atoms,
        )

        # 归一化坐标
        apo_dataset = NormalizeDataset(dataset, "pocket1_coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket2_coordinates")

        # 处理第一个口袋的特征
        src_pocket1_dataset = KeyDataset(apo_dataset, "pocket1_atoms")
        pocket1_len_dataset = LengthDataset(src_pocket1_dataset)
        src_pocket1_dataset = TokenizeDataset(
            src_pocket1_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket1_dataset = KeyDataset(apo_dataset, "pocket1_coordinates")
        src_pocket1_dataset = PrependAndAppend(
            src_pocket1_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket1_edge_type = EdgeTypeDataset(
            src_pocket1_dataset, len(self.pocket_dictionary)
        )
        coord_pocket1_dataset = FromNumpyDataset(coord_pocket1_dataset)
        distance_pocket1_dataset = DistanceDataset(coord_pocket1_dataset)
        coord_pocket1_dataset = PrependAndAppend(coord_pocket1_dataset, 0.0, 0.0)
        distance_pocket1_dataset = PrependAndAppend2DDataset(
            distance_pocket1_dataset, 0.0
        )

        # 处理第二个口袋的特征
        src_pocket2_dataset = KeyDataset(apo_dataset, "pocket2_atoms")
        pocket2_len_dataset = LengthDataset(src_pocket2_dataset)
        src_pocket2_dataset = TokenizeDataset(
            src_pocket2_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket2_dataset = KeyDataset(apo_dataset, "pocket2_coordinates")
        src_pocket2_dataset = PrependAndAppend(
            src_pocket2_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket2_edge_type = EdgeTypeDataset(
            src_pocket2_dataset, len(self.pocket_dictionary)
        )
        coord_pocket2_dataset = FromNumpyDataset(coord_pocket2_dataset)
        distance_pocket2_dataset = DistanceDataset(coord_pocket2_dataset)
        coord_pocket2_dataset = PrependAndAppend(coord_pocket2_dataset, 0.0, 0.0)
        distance_pocket2_dataset = PrependAndAppend2DDataset(
            distance_pocket2_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    # 第一个口袋的输入
                    "pocket1_src_tokens": RightPadDataset(
                        src_pocket1_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket1_src_distance": RightPadDataset2D(
                        distance_pocket1_dataset,
                        pad_idx=0,
                    ),
                    "pocket1_src_edge_type": RightPadDataset2D(
                        pocket1_edge_type,
                        pad_idx=0,
                    ),
                    "pocket1_src_coord": RightPadDatasetCoord(
                        coord_pocket1_dataset,
                        pad_idx=0,
                    ),
                    "pocket1_len": RawArrayDataset(pocket1_len_dataset),
                    
                    # 第二个口袋的输入
                    "pocket2_src_tokens": RightPadDataset(
                        src_pocket2_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket2_src_distance": RightPadDataset2D(
                        distance_pocket2_dataset,
                        pad_idx=0,
                    ),
                    "pocket2_src_edge_type": RightPadDataset2D(
                        pocket2_edge_type,
                        pad_idx=0,
                    ),
                    "pocket2_src_coord": RightPadDatasetCoord(
                        coord_pocket2_dataset,
                        pad_idx=0,
                    ),
                    "pocket2_len": RawArrayDataset(pocket2_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "pocket1_name": RawArrayDataset(pocket1_dataset),
                "pocket2_name": RawArrayDataset(pocket2_dataset),
            },
        )
        
        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_pocket1_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        
        if args.finetune_pocket1_model is not None:
            print("load pretrain pocket1 model weight from...", args.finetune_pocket1_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket1_model,
            )
            model.pocket1_model.load_state_dict(state["model"], strict=False)
            
        if args.finetune_pocket2_model is not None:
            print("load pretrain pocket2 model weight from...", args.finetune_pocket2_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket2_model,
            )
            model.pocket2_model.load_state_dict(state["model"], strict=False)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ): # 这里的sample是取完batch后的
        """训练步骤"""
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output

    def dynamic_crop_batch_pockets(self, sample):
        """
        动态裁剪batch内的pocket，使得同一个batch内的所有pocket1长度一致，所有pocket2长度一致
        这可以减少padding开销，提高训练效率
        """
        # 检查是否启用了动态批量裁剪功能
        if not getattr(self.args, 'dynamic_batch_crop', False):
            return sample
            
        if "net_input" not in sample:
            return sample
            
        net_input = sample["net_input"]
        
        # 获取batch内所有pocket1和pocket2的实际长度
        if "pocket1_len" in net_input and "pocket2_len" in net_input:
            pocket1_lens = net_input["pocket1_len"].cpu().numpy()
            pocket2_lens = net_input["pocket2_len"].cpu().numpy()
            
            # 找到batch内pocket1和pocket2的最小长度（排除padding）
            min_pocket1_len = int(min(pocket1_lens))
            min_pocket2_len = int(min(pocket2_lens))
            
            # 确保最小长度不为0，并且有意义
            min_pocket1_len = max(min_pocket1_len, 2)  # 至少保留[CLS]和[SEP] token
            min_pocket2_len = max(min_pocket2_len, 2)
            
            # 使用CroppingPocketDataset的裁剪逻辑重新裁剪pocket1
            if min_pocket1_len > 2 and min_pocket1_len < net_input["pocket1_src_tokens"].shape[1]:
                self._crop_pocket_tensors(net_input, "pocket1", min_pocket1_len - 2, batch_log_dir=self.batch_log_dir)  # 减去[CLS]和[SEP]

            # 使用CroppingPocketDataset的裁剪逻辑重新裁剪pocket2  
            if min_pocket2_len > 2 and min_pocket2_len < net_input["pocket2_src_tokens"].shape[1]:
                self._crop_pocket_tensors(net_input, "pocket2", min_pocket2_len - 2, batch_log_dir=self.batch_log_dir)  # 减去[CLS]和[SEP]
        
        return sample

    def _crop_pocket_tensors(self, net_input, pocket_prefix, max_atoms, batch_log_dir=None):
        """
        使用数据处理pipeline的逻辑裁剪pocket相关的tensor
        
        Args:
            net_input: 网络输入数据
            pocket_prefix: pocket前缀 ("pocket1" 或 "pocket2")
            max_atoms: 最大原子数（不包括[CLS]和[SEP]）
            batch_log_dir: batch长度日志保存目录（可选）
        """
        from unimol.data.cropping_dataset import crop_pocket_atoms

        # 获取相关tensor
        src_tokens = net_input[f"{pocket_prefix}_src_tokens"]
        src_coord = net_input[f"{pocket_prefix}_src_coord"]

        batch_size = src_tokens.shape[0]

        # 为每个样本单独处理，创建新的数据
        new_atoms_list = []
        new_coords_list = []
        new_lens = []

        # 记录当前batch的pocket长度到文件（可选参数）
        if batch_log_dir is not None:
            os.makedirs(batch_log_dir, exist_ok=True)
            log_file = os.path.join(batch_log_dir, "batch_cropping_stats.tsv")

            # 统计裁剪前的长度
            pre_lens = net_input[f"{pocket_prefix}_len"].cpu().numpy()
            pre_min = int(np.min(pre_lens))
            pre_max = int(np.max(pre_lens))
            pre_mean = float(np.mean(pre_lens))

            # 先准备一行，后续补充裁剪后的长度
            log_row = [pocket_prefix, "pre", str(pre_min), str(pre_max), f"{pre_mean:.2f}"]

        for i in range(batch_size):
            # 获取当前样本的实际长度
            current_len = net_input[f"{pocket_prefix}_len"][i].item()

            # 不再单独写入每个样本长度

            # 确保长度至少为2（[CLS] + [SEP]）
            if current_len < 2:
                current_len = 2

            # 提取当前样本的数据，使用实际长度而不是固定索引
            # 排除[CLS](第0个)和[SEP](最后一个)
            if current_len > 2:
                current_tokens = src_tokens[i, 1:current_len-1]  # 排除[CLS]和[SEP]
                current_coord = src_coord[i, 1:current_len-1, :]  # 排除对应的坐标
            else:
                # 如果长度不足，创建空的tensor
                current_tokens = torch.tensor([], dtype=src_tokens.dtype, device=src_tokens.device)
                current_coord = torch.zeros((0, 3), dtype=src_coord.dtype, device=src_coord.device)

            # 找到实际有效的原子（非padding）
            if len(current_tokens) > 0:
                valid_mask = current_tokens != self.pocket_dictionary.pad()
                if valid_mask.sum() == 0:
                    # 如果没有有效原子，创建最小数据
                    new_atoms_list.append(np.array([4, 5]))  # 默认C, N原子
                    new_coords_list.append(np.zeros((2, 3)))
                    new_lens.append(4)  # 2个原子 + [CLS] + [SEP]
                    continue

                valid_tokens = current_tokens[valid_mask]
                valid_coord = current_coord[valid_mask]
            else:
                # 如果没有有效原子，创建最小数据
                new_atoms_list.append(np.array([4, 5]))  # 默认C, N原子
                new_coords_list.append(np.zeros((2, 3)))
                new_lens.append(4)  # 2个原子 + [CLS] + [SEP]
                continue

            # 使用CroppingPocketDataset的裁剪逻辑
            if len(valid_tokens) > max_atoms:
                # 转换为numpy进行裁剪
                atoms_np = valid_tokens.cpu().numpy()
                coord_np = valid_coord.cpu().numpy()

                cropped_atoms, cropped_coord, selected_indices = crop_pocket_atoms(
                    atoms_np, coord_np, max_atoms, self.seed, i
                )
            else:
                cropped_atoms = valid_tokens[:max_atoms].cpu().numpy()
                cropped_coord = valid_coord[:max_atoms].cpu().numpy()

            new_atoms_list.append(cropped_atoms)
            new_coords_list.append(cropped_coord)
            new_lens.append(len(cropped_atoms) + 2)  # +2 for [CLS] and [SEP]

        # 换行，分隔不同batch（可选参数）
        # 统计裁剪后的长度
        if batch_log_dir is not None:
            post_lens = np.array(new_lens)
            post_min = int(np.min(post_lens))
            post_max = int(np.max(post_lens))
            post_mean = float(np.mean(post_lens))

            # 写入一行，格式：pocket_prefix\tpre\tmin\tmax\tmean\tpost\tmin\tmax\tmean\n
            with open(log_file, "a") as f_log:
                f_log.write(f"{pocket_prefix}\t{pre_min}\t{pre_max}\t{pre_mean:.2f}\t{post_min}\t{post_max}\t{post_mean:.2f}\n")

        # 现在按照标准pipeline重新处理这些数据
        if not new_atoms_list:
            # 如果没有有效数据，直接返回不做任何修改
            return

        original_device = src_tokens.device
        processed_data = self._process_pocket_data_pipeline(
            new_atoms_list, new_coords_list, max_atoms + 2, device=original_device
        )

        # 更新net_input
        net_input[f"{pocket_prefix}_src_tokens"] = processed_data["tokens"]
        net_input[f"{pocket_prefix}_src_coord"] = processed_data["coordinates"]
        net_input[f"{pocket_prefix}_src_distance"] = processed_data["distances"]
        net_input[f"{pocket_prefix}_src_edge_type"] = processed_data["edge_types"]
        net_input[f"{pocket_prefix}_len"] = torch.tensor(new_lens, 
                                                        device=net_input[f"{pocket_prefix}_len"].device)
    
    def _process_pocket_data_pipeline(self, atoms_list, coords_list, max_seq_len, device=None):
        """
        按照标准数据处理pipeline处理pocket数据
        
        Args:
            atoms_list: 每个样本的原子类型列表
            coords_list: 每个样本的坐标列表
            max_seq_len: 最大序列长度（包括特殊token）
            device: 目标设备，如果为None则使用CPU
        
        Returns:
            dict: 包含处理后的tokens, coordinates, distances, edge_types
        """
        
        batch_size = len(atoms_list)
        
        # 1. 创建临时数据结构来模拟dataset
        temp_data = []
        for i, (atoms, coords) in enumerate(zip(atoms_list, coords_list)):
            temp_data.append({
                'atoms': atoms,
                'coordinates': coords.astype(np.float32)
            })
        
        # 2. 模拟TokenizeDataset的处理
        def tokenize_atoms(atoms):
            # 将原子类型转换为词典索引（如果需要）
            return atoms
        
        # 3. 为每个样本处理数据，按照pipeline逻辑
        tokens_list = []
        coords_list_processed = []
        distances_list = []
        edge_types_list = []
        
        for i, data in enumerate(temp_data):
            atoms = data['atoms']
            coords = data['coordinates']
            
            # 3.1 Tokenize (如果原子已经是token索引则跳过)
            tokens = tokenize_atoms(atoms)
            
            # 3.2 PrependAndAppend tokens ([CLS] + atoms + [SEP])
            tokens_with_special = np.concatenate([
                [self.pocket_dictionary.bos()], 
                tokens, 
                [self.pocket_dictionary.eos()]
            ])
            
            # 3.3 PrependAndAppend coordinates (0.0 + coords + 0.0)
            coords_with_special = np.concatenate([
                np.zeros((1, 3)), 
                coords, 
                np.zeros((1, 3))
            ])
            
            # 3.4 计算距离矩阵
            distance_matrix = np.linalg.norm(
                coords_with_special[:, None, :] - coords_with_special[None, :, :], 
                axis=2
            )
            
            # 3.5 计算边类型矩阵
            vocab_size = len(self.pocket_dictionary)
            edge_type_matrix = (tokens_with_special[:, None] * vocab_size + 
                              tokens_with_special[None, :])
            
            # 3.6 Padding到最大长度
            padded_tokens = np.full(max_seq_len, self.pocket_dictionary.pad(), dtype=np.int64)
            padded_coords = np.zeros((max_seq_len, 3), dtype=np.float32)
            padded_distances = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
            padded_edge_types = np.zeros((max_seq_len, max_seq_len), dtype=np.int64)
            
            seq_len = len(tokens_with_special)
            padded_tokens[:seq_len] = tokens_with_special
            padded_coords[:seq_len] = coords_with_special
            padded_distances[:seq_len, :seq_len] = distance_matrix
            padded_edge_types[:seq_len, :seq_len] = edge_type_matrix
            
            tokens_list.append(padded_tokens)
            coords_list_processed.append(padded_coords)
            distances_list.append(padded_distances)
            edge_types_list.append(padded_edge_types)
        
        # 4. 转换为tensor，使用指定的设备
        if device is None:
            device = 'cpu'
        
        return {
            "tokens": torch.tensor(np.stack(tokens_list), dtype=torch.long, device=device),
            "coordinates": torch.tensor(np.stack(coords_list_processed), dtype=torch.float, device=device),
            "distances": torch.tensor(np.stack(distances_list), dtype=torch.float, device=device),
            "edge_types": torch.tensor(np.stack(edge_types_list), dtype=torch.long, device=device)
        }


    def encode_pockets(self, model, data_path, emb_dir, pocket_index, **kwargs):
        """编码口袋表示并缓存"""
        cache_path = os.path.join(emb_dir, data_path.split("/")[-1] + ".pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached pocket embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                pocket_reps, pocket_names = pickle.load(f)
            return pocket_reps, pocket_names

        print(f"Encoding pockets from {data_path}")
        pocket_dataset = self.load_single_pocket_dataset(data_path)
        pocket_reps = []
        pocket_names = []
        bsz = 32
        
        pocket_data = torch.utils.data.DataLoader(
            pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater
        )
        
        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            dist = sample["net_input"]["pocket_src_distance"]
            et = sample["net_input"]["pocket_src_edge_type"]
            st = sample["net_input"]["pocket_src_tokens"]
            
            if pocket_index == "pocket1":
                pocket_padding_mask = st.eq(model.pocket1_model.padding_idx)
                pocket_x = model.pocket1_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket1_model.gbf(dist, et)
                gbf_result = model.pocket1_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket1_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                pocket_emb = model.pocket1_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
                pocket_names.extend(sample["pocket_name"])
            elif pocket_index == "pocket2":
                pocket_padding_mask = st.eq(model.pocket2_model.padding_idx)
                pocket_x = model.pocket2_model.embed_tokens(st)
                n_node = dist.size(-1)
                gbf_feature = model.pocket2_model.gbf(dist, et)
                gbf_result = model.pocket2_model.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                pocket_outputs = model.pocket2_model.encoder(
                    pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                )
                pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                pocket_emb = model.pocket2_project(pocket_encoder_rep)
                pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                pocket_emb = pocket_emb.detach().cpu().numpy()
                pocket_reps.append(pocket_emb)
                pocket_names.extend(sample["pocket_name"])
            else:
                raise ValueError("Invalid pocket index. Use 'pocket1' or 'pocket2'.")

        pocket_reps = np.concatenate(pocket_reps, axis=0)

        # 保存缓存
        os.makedirs(emb_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump([pocket_reps, pocket_names], f)

        return pocket_reps, pocket_names

    def retrieve_pockets(self, model, query_pocket_path, target_pocket_path, emb_dir, k, retrieve_type, **kwargs):
        """
        使用一个pocket2(receptor)去检索另一个pocket1(ligand)库中最相似的pockets
        Args:
            model: 训练好的模型
            query_pocket_path: 查询pocket的数据路径
            target_pocket_path: 目标pocket库的数据路径
            emb_dir: 嵌入向量缓存目录
            k: 返回top-k个结果
        Returns:
            top_k_names: 最相似的k个pocket名称列表
            top_k_scores: 对应的相似度分数
        """
        
        if retrieve_type == "21": #用2搜1
            target_type = "pocket1"
            query_type = "pocket2"
        elif retrieve_type == "12":
            target_type = "pocket2"
            query_type = "pocket1"
        else:
            raise ValueError("retrieve_type must be '21' or '12'.")
        
        os.makedirs(emb_dir, exist_ok=True)        
        
        # 编码目标pocket库
        target_pocket_reps, target_pocket_names = self.encode_pockets(model, target_pocket_path, emb_dir, target_type)
        
        # 编码查询pocket
        query_pocket_dataset = self.load_single_pocket_dataset(query_pocket_path)
        query_pocket_data = torch.utils.data.DataLoader(
            query_pocket_dataset, batch_size=16, collate_fn=query_pocket_dataset.collater
        )
        
        query_pocket_reps = []
        query_pocket_names = []
        
        model.eval()
        with torch.no_grad():
            for _, sample in enumerate(tqdm(query_pocket_data, desc="Encoding query pockets")):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket_src_distance"]
                et = sample["net_input"]["pocket_src_edge_type"]
                st = sample["net_input"]["pocket_src_tokens"]
                
                if query_type == "pocket1":
                    # 使用pocket1_model编码查询pocket
                    pocket_padding_mask = st.eq(model.pocket1_model.padding_idx)
                    pocket_x = model.pocket1_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.pocket1_model.gbf(dist, et)
                    gbf_result = model.pocket1_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    
                    pocket_outputs = model.pocket1_model.encoder(
                        pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                    )
                    pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                    pocket_emb = model.pocket1_project(pocket_encoder_rep)
                    pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                    pocket_emb = pocket_emb.detach().cpu().numpy()
                elif query_type == "pocket2":
                    # 使用pocket2_model编码查询pocket
                    pocket_padding_mask = st.eq(model.pocket2_model.padding_idx)
                    pocket_x = model.pocket2_model.embed_tokens(st)
                    n_node = dist.size(-1)
                    gbf_feature = model.pocket2_model.gbf(dist, et)
                    gbf_result = model.pocket2_model.gbf_proj(gbf_feature)
                    graph_attn_bias = gbf_result
                    graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                    graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                    
                    pocket_outputs = model.pocket2_model.encoder(
                        pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
                    )
                    pocket_encoder_rep = pocket_outputs[0][:, 0, :]
                    pocket_emb = model.pocket2_project(pocket_encoder_rep)
                    pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
                    pocket_emb = pocket_emb.detach().cpu().numpy()
                else:
                    raise ValueError("Invalid query pocket type. Use 'pocket1' or 'pocket2'.")
                
                query_pocket_reps.append(pocket_emb)
                query_pocket_names.extend(sample["pocket_name"])
        
        query_pocket_reps = np.concatenate(query_pocket_reps, axis=0)
        
        # 计算相似度矩阵 (n_query x n_target)
        similarity_matrix = query_pocket_reps @ target_pocket_reps.T
        
        # 为每个query返回top-k结果
        all_results = []
        for i, query_name in enumerate(query_pocket_names):
            query_similarities = similarity_matrix[i]  # 第i个query的所有相似度
            
            # 获取top k结果
            top_k_indices = np.argsort(query_similarities)[::-1][:k]
            top_k_names = [target_pocket_names[idx] for idx in top_k_indices]
            top_k_scores = query_similarities[top_k_indices]
            
            result = {
                'query_name': query_name,
                'top_k_names': top_k_names,
                'top_k_scores': top_k_scores
            }
            all_results.append(result)

        
        return all_results
    
    
    def test_outer(self, data_path, model, **kwargs):
        
        # 测试单个数据集
        auc, bedroc, ef, re, positive_counts = self.test_inner(data_path, model)
        
        # 打印结果
        print("=" * 50)
        print("Test Results")
        print("=" * 50)
        print(f"AUC: {auc:.4f}")
        print(f"BEDROC: {bedroc:.4f}")
        print()
        
        print("Enrichment Factor (EF) Results:")
        print("-" * 30)
        for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
            ef_value = ef[ratio]
            pos_count = positive_counts[ratio]
            total_selected = positive_counts[f"{ratio}_total"]
            print(f"EF @ {float(ratio)*100:4.1f}%: {ef_value:6.2f} (found {pos_count:3d} positives out of {total_selected:4d} selected)")
        
        print()
        print("Recall Enhancement (RE) Results:")
        print("-" * 30)
        for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
            re_value = re[ratio]
            print(f"RE @ {float(ratio)*100:4.1f}%: {re_value:6.2f}")
        
        print("=" * 50)

        return 
    
    def test_inner(self, data_path, model, **kwargs):
        model.eval()
        
        dataset = self.load_test_dataset(data_path)
        num_data = len(dataset)
        bsz=4
        print(num_data//bsz)
        score_list = []
        labels = []
        names = []
        
        data = torch.utils.data.DataLoader(dataset, batch_size=bsz, collate_fn=dataset.collater)
        
        with torch.no_grad():
            for _, sample in enumerate(tqdm(data)):
                sample = unicore.utils.move_to_cuda(sample)
                dist = sample["net_input"]["pocket1_src_distance"]
                et = sample["net_input"]["pocket1_src_edge_type"]
                st = sample["net_input"]["pocket1_src_tokens"]
                pocket1_padding_mask = st.eq(model.pocket1_model.padding_idx)
                pocket1_x = model.pocket1_model.embed_tokens(st)
                
                pocket1_n_node = dist.size(-1)
                pocket1_gbf_feature = model.pocket1_model.gbf(dist, et)
                pocket1_gbf_result = model.pocket1_model.gbf_proj(pocket1_gbf_feature)
                
                pocket1_graph_attn_bias = pocket1_gbf_result
                pocket1_graph_attn_bias = pocket1_graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                pocket1_graph_attn_bias = pocket1_graph_attn_bias.view(-1, pocket1_n_node, pocket1_n_node)
                
                pocket1_outputs = model.pocket1_model.encoder(
                    pocket1_x, padding_mask=pocket1_padding_mask, attn_mask=pocket1_graph_attn_bias
                )
                pocket1_rep = pocket1_outputs[0][:, 0, :]
                
                pocket1_emb = model.pocket1_project(pocket1_rep)
                pocket1_emb = pocket1_emb / pocket1_emb.norm(dim=1, keepdim=True)
                pocket1_emb = pocket1_emb.detach().cpu().numpy()

                # pocket2
                dist = sample["net_input"]["pocket2_src_distance"]
                et = sample["net_input"]["pocket2_src_edge_type"]
                st = sample["net_input"]["pocket2_src_tokens"]
                pocket2_padding_mask = st.eq(model.pocket2_model.padding_idx)
                pocket2_x = model.pocket2_model.embed_tokens(st)
                pocket2_n_node = dist.size(-1)
                pocket2_gbf_feature = model.pocket2_model.gbf(dist, et)
                pocket2_gbf_result = model.pocket2_model.gbf_proj(pocket2_gbf_feature)
                pocket2_graph_attn_bias = pocket2_gbf_result
                pocket2_graph_attn_bias = pocket2_graph_attn_bias.permute(0,3, 1, 2).contiguous()
                pocket2_graph_attn_bias = pocket2_graph_attn_bias.view(-1, pocket2_n_node, pocket2_n_node)
                pocket2_outputs = model.pocket2_model.encoder(
                    pocket2_x, padding_mask=pocket2_padding_mask, attn_mask=pocket2_graph_attn_bias
                )
                pocket2_rep = pocket2_outputs[0][:, 0, :]
                pocket2_emb = model.pocket2_project(pocket2_rep)
                pocket2_emb = pocket2_emb / pocket2_emb.norm(dim=1, keepdim=True)
                pocket2_emb = pocket2_emb.detach().cpu().numpy()
                
                scores = np.sum(pocket1_emb * pocket2_emb, axis=1)
                
                score_list.extend(scores.tolist())
                labels.extend(sample["target"]["finetune_target"].detach().cpu().numpy().tolist())
                names.extend(sample["pocket1_name"])
                
                # 打印label和对应的name和score (可选，调试用)
                # for i in range(len(sample["target"]["finetune_target"])):
                #     print(f"Label: {sample['target']['finetune_target'][i]}, Name: {sample['pocket1_name'][i]}, Score: {scores[i]}")
                # 
                # pocket2_lens = sample["net_input"]["pocket2_len"].detach().cpu().numpy()
                # pocket1_lens = sample["net_input"]["pocket1_len"].detach().cpu().numpy()
                # 输出每个对象的 score 和 pocket2 残基数
                # 输出到文件
                # with open("/data/private/ly/CLIP_test/bcma/test/same_pocket1_score.txt", "a") as f:
                #     for i in range(len(scores)):
                #         line = f"{scores[i]:.4f}\t{pocket1_lens[i]}\t{pocket2_lens[i]}\t{sample['pocket1_name'][i]}\t{sample['pocket2_name'][i]}"
                #         print(line)
                #         f.write(line + "\n")
                
            
        print(score_list)    
        auc, bedroc, ef_list, re_list, positive_counts = cal_metrics(labels, score_list, 80.5)
        
        # 打印数据集基本信息
        print(f"Dataset: {data_path}")
        print(f"Total samples: {positive_counts['total_samples']}")
        print(f"Total positives: {positive_counts['total_positives']}")
        print(f"Positive ratio: {positive_counts['total_positives']/positive_counts['total_samples']:.4f}")
        
        return auc, bedroc, ef_list, re_list, positive_counts
    
    def load_test_dataset(self, data_path, **kwargs):
        """Load pocket-pocket paired dataset.
        Expected data format:
        'pocket1', 'pocket1_atoms', 'pocket1_coordinates', 
        'pocket2', 'pocket2_atoms', 'pocket2_coordinates', 'label'
        """
        dataset = LMDBDataset(data_path)
        
        dataset = PeptideAffinityDataset(
            dataset,
            self.args.seed,
            "pocket1_atoms",
            "pocket1_coordinates",
            "pocket2_atoms",
            "pocket2_coordinates",
            "label",
        )
        tgt_dataset = KeyDataset(dataset, "affinity")
        pocket1_dataset = KeyDataset(dataset, "pocket1")
        pocket2_dataset = KeyDataset(dataset, "pocket2")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # 处理第一个口袋
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket1_atoms",
            "pocket1_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket1_atoms",
            "pocket1_coordinates",
            self.args.max_pocket_atoms,
        )

        # 处理第二个口袋
        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket2_atoms",
            "pocket2_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket2_atoms",
            "pocket2_coordinates",
            self.args.max_pocket_atoms,
        )

        # 归一化坐标
        apo_dataset = NormalizeDataset(dataset, "pocket1_coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket2_coordinates")

        # 处理第一个口袋的特征
        src_pocket1_dataset = KeyDataset(apo_dataset, "pocket1_atoms")
        pocket1_len_dataset = LengthDataset(src_pocket1_dataset)
        src_pocket1_dataset = TokenizeDataset(
            src_pocket1_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket1_dataset = KeyDataset(apo_dataset, "pocket1_coordinates")
        src_pocket1_dataset = PrependAndAppend(
            src_pocket1_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket1_edge_type = EdgeTypeDataset(
            src_pocket1_dataset, len(self.pocket_dictionary)
        )
        coord_pocket1_dataset = FromNumpyDataset(coord_pocket1_dataset)
        distance_pocket1_dataset = DistanceDataset(coord_pocket1_dataset)
        coord_pocket1_dataset = PrependAndAppend(coord_pocket1_dataset, 0.0, 0.0)
        distance_pocket1_dataset = PrependAndAppend2DDataset(
            distance_pocket1_dataset, 0.0
        )

        # 处理第二个口袋的特征
        src_pocket2_dataset = KeyDataset(apo_dataset, "pocket2_atoms")
        pocket2_len_dataset = LengthDataset(src_pocket2_dataset)
        src_pocket2_dataset = TokenizeDataset(
            src_pocket2_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket2_dataset = KeyDataset(apo_dataset, "pocket2_coordinates")
        src_pocket2_dataset = PrependAndAppend(
            src_pocket2_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket2_edge_type = EdgeTypeDataset(
            src_pocket2_dataset, len(self.pocket_dictionary)
        )
        coord_pocket2_dataset = FromNumpyDataset(coord_pocket2_dataset)
        distance_pocket2_dataset = DistanceDataset(coord_pocket2_dataset)
        coord_pocket2_dataset = PrependAndAppend(coord_pocket2_dataset, 0.0, 0.0)
        distance_pocket2_dataset = PrependAndAppend2DDataset(
            distance_pocket2_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    # 第一个口袋的输入
                    "pocket1_src_tokens": RightPadDataset(
                        src_pocket1_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket1_src_distance": RightPadDataset2D(
                        distance_pocket1_dataset,
                        pad_idx=0,
                    ),
                    "pocket1_src_edge_type": RightPadDataset2D(
                        pocket1_edge_type,
                        pad_idx=0,
                    ),
                    "pocket1_src_coord": RightPadDatasetCoord(
                        coord_pocket1_dataset,
                        pad_idx=0,
                    ),
                    "pocket1_len": RawArrayDataset(pocket1_len_dataset),
                    
                    # 第二个口袋的输入
                    "pocket2_src_tokens": RightPadDataset(
                        src_pocket2_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket2_src_distance": RightPadDataset2D(
                        distance_pocket2_dataset,
                        pad_idx=0,
                    ),
                    "pocket2_src_edge_type": RightPadDataset2D(
                        pocket2_edge_type,
                        pad_idx=0,
                    ),
                    "pocket2_src_coord": RightPadDatasetCoord(
                        coord_pocket2_dataset,
                        pad_idx=0,
                    ),
                    "pocket2_len": RawArrayDataset(pocket2_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "pocket1_name": RawArrayDataset(pocket1_dataset),
                "pocket2_name": RawArrayDataset(pocket2_dataset),
            },
        )
        
        return nest_dataset