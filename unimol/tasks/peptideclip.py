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
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
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
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index)*0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list


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

    def load_single_pocket_dataset(self, data_path, **kwargs):
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

        nest_dataset = NestedDictionaryDataset(
            {
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
            },
        )
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
        
        # 计算相似度矩阵 (query_pockets x target_pockets)
        res = query_pocket_reps @ target_pocket_reps.T
        res = res.max(axis=0)  # 取每个target pocket的最高相似度分数
        
        # 获取top k结果
        top_k = np.argsort(res)[::-1][:k]
        
        # 返回名称和分数
        return [target_pocket_names[i] for i in top_k], res[top_k]