#!/usr/bin/env python3
"""
PeptideCLIP Inference Script

This script extracts the test logic from the PeptideCLIP task to perform inference
on pocket-pocket pairs. Given a model and input data, it processes the input data
and returns pocket1 and pocket2 representations along with the final similarity scores.

Usage:
    python inference.py --model_path <model_checkpoint> --data_path <test_data.lmdb> [options]
"""

import sys
import os

# Add the parent directory to Python path to import unimol
# Current file is in /path/to/PeptideCLIP/unimol/inference.py
# We need to add /path/to/PeptideCLIP to sys.path to import unimol
current_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/PeptideCLIP/unimol
parent_dir = os.path.dirname(current_dir)  # /path/to/PeptideCLIP
sys.path.insert(0, parent_dir)

import argparse
import logging
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Unicore and Unimol imports
import unicore
from unicore import checkpoint_utils
from unicore.data import (
    AppendTokenDataset, Dictionary, FromNumpyDataset,
    NestedDictionaryDataset, PrependTokenDataset, RawArrayDataset, 
    LMDBDataset, RawLabelDataset, RightPadDataset, RightPadDataset2D, 
    TokenizeDataset, data_utils
)

from unimol.data import (
    PeptideAffinityDataset, CroppingPocketDataset,
    DistanceDataset, EdgeTypeDataset, KeyDataset, LengthDataset,
    NormalizeDataset, PrependAndAppend2DDataset, 
    RemoveHydrogenPocketDataset, RightPadDatasetCoord
)

from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def re_new(y_true, y_score, ratio):
    """Calculate recall enhancement at given ratio."""
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    """Calculate recall enhancement for multiple ratios."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)
    
    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)
    
    return res2


def cal_metrics(y_true, y_score, alpha):
    """Calculate BEDROC score and other metrics for pocket similarity evaluation."""
    y_true_array = np.array(y_true)
    total_samples = len(y_true)
    total_positives = int(np.sum(y_true_array))
    
    # 检查是否所有标签都是0（全是阴性样本）
    if total_positives == 0:
        logger.warning("All samples are negative (label=0). Evaluation metrics will be N/A or 0.")
        
        # 为全阴性情况创建占位符结果
        ratios = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        ef = {}
        positive_counts = {}
        re_list = {}
        
        for ratio in ratios:
            ef[str(ratio)] = 0.0  # 没有正样本，EF为0
            positive_counts[str(ratio)] = 0
            positive_counts[f"{ratio}_total"] = max(1, int(total_samples * ratio))
            re_list[str(ratio)] = 0.0  # 没有正样本，RE为0
        
        # 添加总体统计信息
        positive_counts["total_positives"] = total_positives
        positive_counts["total_samples"] = total_samples
        
        return 0.5, 0.0, ef, re_list, positive_counts  # AUC=0.5 (random), BEDROC=0
    
    # 检查是否所有标签都是1（全是正样本）
    if total_positives == total_samples:
        logger.warning("All samples are positive (label=1). Some evaluation metrics may not be meaningful.")
    
    # 正常情况：有正样本和负样本
    scores = np.expand_dims(y_score, axis=1)
    y_true_expanded = np.expand_dims(y_true_array, axis=1)
    scores = np.concatenate((scores, y_true_expanded), axis=1)
    scores = scores[scores[:, 0].argsort()[::-1]]
    
    try:
        bedroc = CalcBEDROC(scores, 1, 80.5)
    except Exception as e:
        logger.warning(f"Failed to calculate BEDROC: {e}")
        bedroc = 0.0
    
    try:
        auc = CalcAUC(scores, 1)
    except Exception as e:
        logger.warning(f"Failed to calculate AUC: {e}")
        auc = 0.5  # Random classifier AUC
    
    index = np.argsort(y_score)[::-1]
    
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
    
    try:
        re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    except Exception as e:
        logger.warning(f"Failed to calculate recall enhancement: {e}")
        re_list = {str(ratio): 0.0 for ratio in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]}
    
    return auc, bedroc, ef, re_list, positive_counts


class PeptideCLIPInference:
    """PeptideCLIP inference class for pocket-pocket similarity prediction."""
    
    def __init__(self, model_path, dict_path, max_pocket_atoms=256, max_seq_len=512):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            dict_path: Path to the pocket dictionary file
            max_pocket_atoms: Maximum number of atoms in a pocket
            max_seq_len: Maximum sequence length
        """
        self.max_pocket_atoms = max_pocket_atoms
        self.max_seq_len = max_seq_len
        
        # Load dictionary
        self.pocket_dictionary = Dictionary.load(dict_path)
        logger.info(f"Loaded pocket dictionary: {len(self.pocket_dictionary)} types")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        logger.info("Model loaded and set to evaluation mode")
    
    def _load_model(self, model_path):
        """Load the trained model from checkpoint."""
        # Load checkpoint
        state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
        
        # Check if 'args' exists in the checkpoint
        if "args" not in state:
            raise ValueError("No 'args' found in checkpoint. Please ensure you're using a valid PeptideCLIP checkpoint.")
        
        args = state["args"]
        
        # Import tasks after path setup
        from unicore import tasks
        
        # Setup task and build model
        task = tasks.setup_task(args)
        model = task.build_model(args)
        
        # Load model weights
        model.load_state_dict(state["model"], strict=False)
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model moved to CUDA")
        
        return model
    
    def load_dataset(self, data_path):
        """
        Load pocket-pocket paired dataset.
        
        Expected data format:
        'pocket1', 'pocket1_atoms', 'pocket1_coordinates', 
        'pocket2', 'pocket2_atoms', 'pocket2_coordinates', 'label'
        """
        dataset = LMDBDataset(data_path)
        
        dataset = PeptideAffinityDataset(
            dataset,
            42,  # seed
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
            42,  # seed
            "pocket1_atoms",
            "pocket1_coordinates",
            self.max_pocket_atoms,
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
            42,  # seed
            "pocket2_atoms",
            "pocket2_coordinates",
            self.max_pocket_atoms,
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
            max_seq_len=self.max_seq_len,
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
            max_seq_len=self.max_seq_len,
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
    
    def get_pocket_representation(self, sample, pocket_prefix):
        """
        Get representation for a single pocket (pocket1 or pocket2).
        
        Args:
            sample: Input sample containing pocket data
            pocket_prefix: Either "pocket1" or "pocket2"
            
        Returns:
            Normalized pocket embedding
        """
        # Select the appropriate model
        if pocket_prefix == "pocket1":
            pocket_model = self.model.pocket1_model
            project_layer = self.model.pocket1_project
        else:
            pocket_model = self.model.pocket2_model
            project_layer = self.model.pocket2_project
        
        # Extract input features
        dist = sample["net_input"][f"{pocket_prefix}_src_distance"]
        et = sample["net_input"][f"{pocket_prefix}_src_edge_type"]
        st = sample["net_input"][f"{pocket_prefix}_src_tokens"]
        
        # Create padding mask
        padding_mask = st.eq(pocket_model.padding_idx)
        
        # Get token embeddings
        x = pocket_model.embed_tokens(st)
        
        # Calculate graph attention bias
        n_node = dist.size(-1)
        gbf_feature = pocket_model.gbf(dist, et)
        gbf_result = pocket_model.gbf_proj(gbf_feature)
        
        graph_attn_bias = gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        
        # Forward through encoder
        outputs = pocket_model.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        
        # Get [CLS] token representation
        pocket_rep = outputs[0][:, 0, :]
        
        # Project and normalize
        pocket_emb = project_layer(pocket_rep)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=1, keepdim=True)
        
        return pocket_emb
    
    def predict_batch(self, sample):
        """
        Predict similarity scores for a batch of pocket pairs.
        
        Args:
            sample: Batch of input samples
            
        Returns:
            tuple: (pocket1_embeddings, pocket2_embeddings, similarity_scores)
        """
        with torch.no_grad():
            # Get pocket representations
            pocket1_emb = self.get_pocket_representation(sample, "pocket1")
            pocket2_emb = self.get_pocket_representation(sample, "pocket2")
            
            # Calculate similarity scores (cosine similarity)
            scores = torch.sum(pocket1_emb * pocket2_emb, dim=1)
            
            return pocket1_emb, pocket2_emb, scores
    
    def evaluate_dataset(self, data_path, batch_size=4, save_results=None):
        """
        Evaluate the model on a complete dataset.
        
        Args:
            data_path: Path to the LMDB dataset
            batch_size: Batch size for inference
            save_results: Optional path to save detailed results
            
        Returns:
            Dictionary containing evaluation metrics and results
        """
        logger.info(f"Loading dataset from: {data_path}")
        dataset = self.load_dataset(data_path)
        num_data = len(dataset)
        logger.info(f"Dataset loaded: {num_data} samples")
        
        # Prepare data loader
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=dataset.collater
        )
        
        # Storage for results
        all_scores = []
        all_labels = []
        all_pocket1_names = []
        all_pocket2_names = []
        all_pocket1_embs = []
        all_pocket2_embs = []
        
        logger.info("Starting inference...")
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(data_loader, desc="Processing batches")):
                # Move to CUDA if available
                sample = unicore.utils.move_to_cuda(sample)
                
                # Get predictions
                pocket1_emb, pocket2_emb, scores = self.predict_batch(sample)
                
                # Store results
                all_scores.extend(scores.detach().cpu().numpy().tolist())
                all_labels.extend(sample["target"]["finetune_target"].detach().cpu().numpy().tolist())
                all_pocket1_names.extend(sample["pocket1_name"])
                all_pocket2_names.extend(sample["pocket2_name"])
                all_pocket1_embs.append(pocket1_emb.detach().cpu().numpy())
                all_pocket2_embs.append(pocket2_emb.detach().cpu().numpy())
        
        logger.info("Inference completed. Calculating metrics...")
        
        # Calculate metrics
        auc, bedroc, ef_list, re_list, positive_counts = cal_metrics(all_labels, all_scores, 80.5)
        
        # Prepare results
        results = {
            "auc": auc,
            "bedroc": bedroc,
            "enrichment_factors": ef_list,
            "recall_enhancements": re_list,
            "positive_counts": positive_counts,
            "scores": all_scores,
            "labels": all_labels,
            "pocket1_names": all_pocket1_names,
            "pocket2_names": all_pocket2_names,
            "pocket1_embeddings": np.concatenate(all_pocket1_embs, axis=0),
            "pocket2_embeddings": np.concatenate(all_pocket2_embs, axis=0),
        }
        
        # Print summary
        self._print_evaluation_summary(results)
        
        # Save results if requested
        if save_results:
            self._save_results(results, save_results)
            
        return results
    
    def _print_evaluation_summary(self, results):
        """Print evaluation summary."""
        print("=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"AUC: {results['auc']:.4f}")
        print(f"BEDROC: {results['bedroc']:.4f}")
        print()
        
        print("Enrichment Factor (EF) Results:")
        print("-" * 30)
        ef = results['enrichment_factors']
        positive_counts = results['positive_counts']
        for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
            ef_value = ef[ratio]
            pos_count = positive_counts[ratio]
            total_selected = positive_counts[f"{ratio}_total"]
            print(f"EF @ {float(ratio)*100:4.1f}%: {ef_value:6.2f} "
                  f"(found {pos_count:3d} positives out of {total_selected:4d} selected)")
        
        print()
        print("Recall Enhancement (RE) Results:")
        print("-" * 30)
        re = results['recall_enhancements']
        for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
            re_value = re[ratio]
            print(f"RE @ {float(ratio)*100:4.1f}%: {re_value:6.2f}")
        
        print()
        print(f"Dataset Info:")
        print(f"Total samples: {positive_counts['total_samples']}")
        print(f"Total positives: {positive_counts['total_positives']}")
        print(f"Positive ratio: {positive_counts['total_positives']/positive_counts['total_samples']:.4f}")
        print("=" * 50)
    
    def _save_results(self, results, save_path):
        """Save results to separate files: embeddings as CSV and evaluation as text."""
        logger.info(f"Saving results to: {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save embeddings as CSV
        embeddings_path = os.path.join(os.path.dirname(save_path), 'embeddings.csv')
        self._save_embeddings_csv(results, embeddings_path)
        
        # Save evaluation results as text
        evaluation_path = os.path.join(os.path.dirname(save_path), 'evaluation.txt')
        self._save_evaluation_results(results, evaluation_path)
        
        # Keep the original pickle file for complete data if needed
        with open(os.path.join(os.path.dirname(save_path), 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved successfully:")
        logger.info(f"  - Embeddings: {embeddings_path}")
        logger.info(f"  - Evaluation: {evaluation_path}")
        logger.info(f"  - Complete data: {save_path}")
    
    def create_umap_visualization(self, results, save_path=None, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Create UMAP visualization for pocket1 and pocket2 embeddings separately.
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Path to save the UMAP plots (optional)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            random_state: Random state for reproducibility
        """
        try:
            logger.info("Creating UMAP visualizations for pocket1 and pocket2 embeddings...")
            
            # Extract embeddings and labels
            pocket1_embeddings = results['pocket1_embeddings']
            pocket2_embeddings = results['pocket2_embeddings']
            labels = np.array(results['labels'])
            pocket1_names = results['pocket1_names']
            pocket2_names = results['pocket2_names']
            
            # Create UMAP objects
            umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            
            # Fit and transform pocket1 embeddings
            logger.info("Performing UMAP transformation for pocket1 embeddings...")
            pocket1_umap = umap_reducer.fit_transform(pocket1_embeddings)
            
            # Fit and transform pocket2 embeddings with a new reducer instance
            umap_reducer_2 = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            logger.info("Performing UMAP transformation for pocket2 embeddings...")
            pocket2_umap = umap_reducer_2.fit_transform(pocket2_embeddings)
            
            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot pocket1 UMAP
            self._plot_umap(axes[0], pocket1_umap, labels, pocket1_names, "Pocket1 Embeddings UMAP")
            
            # Plot pocket2 UMAP
            self._plot_umap(axes[1], pocket2_umap, labels, pocket2_names, "Pocket2 Embeddings UMAP")
            
            plt.tight_layout()
            
            # Save plot if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"UMAP visualization saved to: {save_path}")
            
            plt.show()
            
            # Return UMAP coordinates for further analysis
            return {
                'pocket1_umap': pocket1_umap,
                'pocket2_umap': pocket2_umap,
                'labels': labels,
                'pocket1_names': pocket1_names,
                'pocket2_names': pocket2_names
            }
            
        except ImportError as e:
            logger.error(f"Missing required packages for UMAP visualization: {e}")
            logger.info("Please install required packages: pip install umap-learn matplotlib seaborn")
            return None
        except Exception as e:
            logger.error(f"Error creating UMAP visualization: {e}")
            return None
    
    def _plot_umap(self, ax, umap_coords, labels, names, title):
        """
        Helper function to plot UMAP coordinates.
        
        Args:
            ax: Matplotlib axis object
            umap_coords: UMAP coordinates (N x 2)
            labels: Labels for coloring points
            names: Names for points (for potential annotations)
            title: Plot title
        """
        # Create color mapping
        unique_labels = np.unique(labels)
        colors = ['red' if label == 1 else 'blue' for label in labels]
        
        # Create scatter plot
        scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                           c=colors, alpha=0.6, s=50)
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=8, label='Positive (Label=1)'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                 markersize=8, label='Negative (Label=0)')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add grid for better visualization
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        positive_count = np.sum(labels == 1)
        total_count = len(labels)
        stats_text = f"Total: {total_count}\nPositive: {positive_count}\nNegative: {total_count - positive_count}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def compare_pocket_distributions(self, results, save_path=None):
        """
        Compare the distributions of pocket1 and pocket2 embeddings.
        
        Args:
            results: Results dictionary from evaluate_dataset
            save_path: Path to save the comparison plots
        """
        try:
            logger.info("Comparing pocket1 and pocket2 embedding distributions...")
            
            # Get UMAP coordinates
            umap_data = self.create_umap_visualization(results, save_path=None)
            if umap_data is None:
                return None
            
            pocket1_umap = umap_data['pocket1_umap']
            pocket2_umap = umap_data['pocket2_umap']
            labels = umap_data['labels']
            
            # Calculate distribution metrics
            comparison_results = self._calculate_distribution_metrics(pocket1_umap, pocket2_umap, labels)
            
            # Create comparison visualization
            self._create_comparison_plots(pocket1_umap, pocket2_umap, labels, comparison_results, save_path)
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing pocket distributions: {e}")
            return None
    
    def _calculate_distribution_metrics(self, pocket1_umap, pocket2_umap, labels):
        """Calculate metrics to compare pocket1 and pocket2 distributions."""
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import wasserstein_distance
        
        results = {}
        
        # Calculate pairwise distances within each pocket type
        pocket1_distances = pdist(pocket1_umap)
        pocket2_distances = pdist(pocket2_umap)
        
        # Calculate mean distances
        results['pocket1_mean_distance'] = np.mean(pocket1_distances)
        results['pocket2_mean_distance'] = np.mean(pocket2_distances)
        
        # Calculate standard deviations
        results['pocket1_std_distance'] = np.std(pocket1_distances)
        results['pocket2_std_distance'] = np.std(pocket2_distances)
        
        # Calculate Wasserstein distance between the two distributions
        try:
            results['wasserstein_distance_x'] = wasserstein_distance(pocket1_umap[:, 0], pocket2_umap[:, 0])
            results['wasserstein_distance_y'] = wasserstein_distance(pocket1_umap[:, 1], pocket2_umap[:, 1])
        except:
            results['wasserstein_distance_x'] = None
            results['wasserstein_distance_y'] = None
        
        # Calculate label-specific metrics
        for label_val in [0, 1]:
            label_mask = labels == label_val
            if np.sum(label_mask) > 1:  # Need at least 2 points for distance calculation
                p1_label = pocket1_umap[label_mask]
                p2_label = pocket2_umap[label_mask]
                
                if len(p1_label) > 1:
                    p1_distances = pdist(p1_label)
                    results[f'pocket1_label_{label_val}_mean_dist'] = np.mean(p1_distances)
                
                if len(p2_label) > 1:
                    p2_distances = pdist(p2_label)
                    results[f'pocket2_label_{label_val}_mean_dist'] = np.mean(p2_distances)
        
        return results
    
    def _create_comparison_plots(self, pocket1_umap, pocket2_umap, labels, metrics, save_path):
        """Create comprehensive comparison plots."""
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Plot 1: Pocket1 UMAP
        self._plot_umap(axes[0, 0], pocket1_umap, labels, None, "Pocket1 Embeddings")
        
        # Plot 2: Pocket2 UMAP
        self._plot_umap(axes[0, 1], pocket2_umap, labels, None, "Pocket2 Embeddings")
        
        # Plot 3: Overlay comparison
        axes[0, 2].scatter(pocket1_umap[:, 0], pocket1_umap[:, 1], 
                          c='red', alpha=0.5, s=30, label='Pocket1')
        axes[0, 2].scatter(pocket2_umap[:, 0], pocket2_umap[:, 1], 
                          c='blue', alpha=0.5, s=30, label='Pocket2')
        axes[0, 2].set_title('Pocket1 vs Pocket2 Overlay', fontsize=14, fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Distance distributions
        from scipy.spatial.distance import pdist
        p1_distances = pdist(pocket1_umap)
        p2_distances = pdist(pocket2_umap)
        
        axes[1, 0].hist(p1_distances, bins=50, alpha=0.7, label='Pocket1', color='red')
        axes[1, 0].hist(p2_distances, bins=50, alpha=0.7, label='Pocket2', color='blue')
        axes[1, 0].set_title('Pairwise Distance Distributions')
        axes[1, 0].set_xlabel('Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Plot 5: Dimension distributions
        axes[1, 1].hist(pocket1_umap[:, 0], bins=30, alpha=0.7, label='Pocket1 X', color='red')
        axes[1, 1].hist(pocket2_umap[:, 0], bins=30, alpha=0.7, label='Pocket2 X', color='blue')
        axes[1, 1].set_title('X Dimension Distributions')
        axes[1, 1].set_xlabel('UMAP Dimension 1')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # Plot 6: Metrics summary
        axes[1, 2].axis('off')
        metrics_text = "Distribution Comparison Metrics:\n\n"
        metrics_text += f"Pocket1 mean distance: {metrics.get('pocket1_mean_distance', 'N/A'):.4f}\n"
        metrics_text += f"Pocket2 mean distance: {metrics.get('pocket2_mean_distance', 'N/A'):.4f}\n\n"
        metrics_text += f"Pocket1 std distance: {metrics.get('pocket1_std_distance', 'N/A'):.4f}\n"
        metrics_text += f"Pocket2 std distance: {metrics.get('pocket2_std_distance', 'N/A'):.4f}\n\n"
        if metrics.get('wasserstein_distance_x') is not None:
            metrics_text += f"Wasserstein distance (X): {metrics['wasserstein_distance_x']:.4f}\n"
            metrics_text += f"Wasserstein distance (Y): {metrics['wasserstein_distance_y']:.4f}\n"
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            comparison_path = save_path.replace('.png', '_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison visualization saved to: {comparison_path}")
        
        plt.show()
    
    def _save_embeddings_csv(self, results, csv_path):
        """Save basic results as CSV file (without embeddings) and embeddings separately."""
        import csv
        
        # Save basic results as CSV (metadata + scores)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['pocket1_name', 'pocket2_name', 'score', 'label'])
            
            # Write data
            for i, pocket1_name in enumerate(results['pocket1_names']):
                pocket2_name = results['pocket2_names'][i]
                score = results['scores'][i]
                label = results['labels'][i]
                
                writer.writerow([pocket1_name, pocket2_name, f'{score:.6f}', label])
        
        embedding_dir = os.path.dirname(csv_path)
        
        # Save as .npz (compressed numpy format)
        embeddings_npz_path = os.path.join(embedding_dir, 'embeddings.npz')
        np.savez_compressed(
            embeddings_npz_path,
            pocket1_embeddings=results['pocket1_embeddings'],
            pocket2_embeddings=results['pocket2_embeddings'],
            pocket1_names=results['pocket1_names'],
            pocket2_names=results['pocket2_names'],
            scores=results['scores'],
            labels=results['labels']
        )
        
        # Also save individual embedding files for easy access
        np.save(os.path.join(embedding_dir, 'pocket1_embeddings.npy'), results['pocket1_embeddings'])
        np.save(os.path.join(embedding_dir, 'pocket2_embeddings.npy'), results['pocket2_embeddings'])
        
        logger.info(f"Basic results saved to: {csv_path}")
        logger.info(f"Embeddings saved to: {embedding_dir}")
        logger.info(f"Compressed embeddings: {embeddings_npz_path}")
    
    def _save_evaluation_results(self, results, txt_path):
        """Save evaluation metrics as text file."""
        with open(txt_path, 'w') as f:
            f.write("PeptideCLIP Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"AUC: {results['auc']:.4f}\n")
            f.write(f"BEDROC: {results['bedroc']:.4f}\n\n")
            
            f.write("Enrichment Factor (EF) Results:\n")
            f.write("-" * 30 + "\n")
            ef = results['enrichment_factors']
            positive_counts = results['positive_counts']
            for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
                ef_value = ef[ratio]
                pos_count = positive_counts[ratio]
                total_selected = positive_counts[f"{ratio}_total"]
                f.write(f"EF @ {float(ratio)*100:4.1f}%: {ef_value:6.2f} "
                       f"(found {pos_count:3d} positives out of {total_selected:4d} selected)\n")
            
            f.write("\nRecall Enhancement (RE) Results:\n")
            f.write("-" * 30 + "\n")
            re = results['recall_enhancements']
            for ratio in ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"]:
                re_value = re[ratio]
                f.write(f"RE @ {float(ratio)*100:4.1f}%: {re_value:6.2f}\n")
            
            f.write(f"\nDataset Info:\n")
            f.write(f"Total samples: {positive_counts['total_samples']}\n")
            f.write(f"Total positives: {positive_counts['total_positives']}\n")
            f.write(f"Positive ratio: {positive_counts['total_positives']/positive_counts['total_samples']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="PeptideCLIP Inference Script")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to test data (LMDB format)")
    parser.add_argument("--dict_path", required=True, help="Path to pocket dictionary file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--max_pocket_atoms", type=int, default=256, help="Maximum pocket atoms")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--save_results", help="Path to save detailed results (optional)")
    parser.add_argument("--output_embeddings", action="store_true", 
                       help="Include embeddings in saved results")
    parser.add_argument("--create_umap", action="store_true",
                       help="Create UMAP visualization for pocket embeddings")
    parser.add_argument("--umap_save_path", help="Path to save UMAP visualization")
    parser.add_argument("--compare_distributions", action="store_true",
                       help="Compare pocket1 and pocket2 embedding distributions")
    parser.add_argument("--umap_n_neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    parser.add_argument("--umap_min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    logger.info("Initializing PeptideCLIP inference engine...")
    inference_engine = PeptideCLIPInference(
        model_path=args.model_path,
        dict_path=args.dict_path,
        max_pocket_atoms=args.max_pocket_atoms,
        max_seq_len=args.max_seq_len
    )
    
    # Evaluate dataset
    results = inference_engine.evaluate_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        save_results=args.save_results,
    )
    
    # Create UMAP visualization if requested
    if args.create_umap:
        logger.info("Creating UMAP visualization...")
        umap_data = inference_engine.create_umap_visualization(
            results=results,
            save_path=args.umap_save_path,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist
        )
        
        if umap_data is not None:
            logger.info("UMAP visualization completed successfully!")
        else:
            logger.warning("UMAP visualization failed. Please check dependencies.")
    
    # Compare distributions if requested
    if args.compare_distributions:
        logger.info("Comparing pocket distributions...")
        comparison_results = inference_engine.compare_pocket_distributions(
            results=results,
            save_path=args.umap_save_path
        )
        
        if comparison_results is not None:
            logger.info("Distribution comparison completed successfully!")
            logger.info("Distribution metrics:")
            for key, value in comparison_results.items():
                if value is not None:
                    logger.info(f"  {key}: {value:.4f}")
        else:
            logger.warning("Distribution comparison failed.")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
