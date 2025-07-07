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
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

    def __init__(self, args, pocket_dictionary):
        super().__init__(args)
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)

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

    def load_receptor_dataset(self, data_path, **kwargs):
        """加载单个口袋数据集，用于检索任务"""
        dataset = LMDBDataset(data_path)
        
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset_dict = {
            "net_input": {
                "pocket_src_tokens": RightPadDataset(
                    src_pocket_dataset,
                    pad_idx=self.pocket_dictionary.pad(),
                ),
                "pocket_src_distance": RightPadDataset2D(
                    distance_pocket_dataset,
                    pad_idx=0,
                ),
                "pocket_src_edge_type": RightPadDataset2D(
                    pocket_edge_type,
                    pad_idx=0,
                ),
                "pocket_src_coord": RightPadDatasetCoord(
                    coord_pocket_dataset,
                    pad_idx=0,
                ),
            },
            "pocket_name": RawArrayDataset(poc_dataset),
            "pocket_len": RawArrayDataset(len_dataset),
        }
        
        nest_dataset = NestedDictionaryDataset(nest_dataset_dict)
        return nest_dataset
    
    def load_peptide_dataset(self, data_path, **kwargs):
        """加载单个口袋数据集，用于检索任务"""
        dataset = LMDBDataset(data_path)
        
        label_dataset = KeyDataset(dataset, "label")
        
        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset_dict = {
            "net_input": {
                "pocket_src_tokens": RightPadDataset(
                    src_pocket_dataset,
                    pad_idx=self.pocket_dictionary.pad(),
                ),
                "pocket_src_distance": RightPadDataset2D(
                    distance_pocket_dataset,
                    pad_idx=0,
                ),
                "pocket_src_edge_type": RightPadDataset2D(
                    pocket_edge_type,
                    pad_idx=0,
                ),
                "pocket_src_coord": RightPadDatasetCoord(
                    coord_pocket_dataset,
                    pad_idx=0,
                ),
            },
            "pocket_name": RawArrayDataset(poc_dataset),
            "target": RawArrayDataset(label_dataset),
            "pocket_len": RawArrayDataset(len_dataset),
        }
        
        nest_dataset = NestedDictionaryDataset(nest_dataset_dict)
        return nest_dataset

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
    ):
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