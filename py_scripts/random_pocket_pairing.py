#!/usr/bin/env python3
"""
Random Pocket Pairing Script

从现有的LMDB数据库中随机选择n个样本，然后随机重新配对pocket1和pocket2，
创建负样本数据（label=0），并生成新的LMDB数据库。

Usage:
    python random_pocket_pairing.py --input_lmdb input.lmdb --output_lmdb output.lmdb --num_samples 1000
"""

import os
import lmdb
import pickle
import numpy as np
import argparse
import random
from tqdm import tqdm
from collections import defaultdict

def read_lmdb_data(lmdb_path):
    """
    读取LMDB数据库中的所有数据
    
    Args:
        lmdb_path: LMDB文件路径
        
    Returns:
        list: 包含所有数据条目的列表
    """
    print(f"Reading data from {lmdb_path}...")
    
    env = lmdb.open(lmdb_path, readonly=True, subdir=False)
    data_list = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, desc="Loading data"):
            data_entry = pickle.loads(value)
            data_list.append(data_entry)
    
    env.close()
    print(f"Loaded {len(data_list)} entries from LMDB")
    return data_list

def extract_pocket_pools(data_list):
    """
    从数据中提取pocket1和pocket2的池子
    
    Args:
        data_list: 数据条目列表
        
    Returns:
        tuple: (pocket1_pool, pocket2_pool) 包含所有pocket信息的列表
    """
    pocket1_pool = []
    pocket2_pool = []
    
    for data_entry in data_list:
        # 提取pocket1信息
        pocket1_info = {
            'name': data_entry['pocket1'],
            'atoms': data_entry['pocket1_atoms'],
            'coordinates': data_entry['pocket1_coordinates']
        }
        pocket1_pool.append(pocket1_info)
        
        # 提取pocket2信息
        pocket2_info = {
            'name': data_entry['pocket2'],
            'atoms': data_entry['pocket2_atoms'],
            'coordinates': data_entry['pocket2_coordinates']
        }
        pocket2_pool.append(pocket2_info)
    
    print(f"Extracted {len(pocket1_pool)} pocket1 entries and {len(pocket2_pool)} pocket2 entries")
    return pocket1_pool, pocket2_pool

def create_positive_samples(original_data, num_samples, seed=42):
    """
    从原始数据中随机抽样，保持原始配对和正样本标签（label=1）
    
    Args:
        original_data: 原始数据列表
        num_samples: 要抽样的样本数量
        seed: 随机种子
        
    Returns:
        list: 抽样的正样本数据列表
    """
    print(f"Sampling {num_samples} positive samples from original data...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保不超过原始数据的数量
    if num_samples > len(original_data):
        print(f"Warning: Requested {num_samples} samples, but only {len(original_data)} available.")
        print(f"Using all {len(original_data)} samples.")
        num_samples = len(original_data)
    
    # 随机抽样
    sampled_indices = random.sample(range(len(original_data)), num_samples)
    
    sampled_data = []
    for i, idx in enumerate(sampled_indices):
        original_entry = original_data[idx]
        
        # 创建新的数据条目，保持原始配对和标签
        sampled_entry = {
            'pocket1': original_entry['pocket1'],  # 保留原始pocket1名字
            'pocket1_atoms': original_entry['pocket1_atoms'].copy(),
            'pocket1_coordinates': original_entry['pocket1_coordinates'].copy(),
            'pocket2': original_entry['pocket2'],  # 保留原始pocket2名字
            'pocket2_atoms': original_entry['pocket2_atoms'].copy(),
            'pocket2_coordinates': original_entry['pocket2_coordinates'].copy(),
            'label': 1,  # 正样本标签
            'sample_index': i,  # 添加抽样索引
            'original_index': idx,  # 添加原始数据索引
            'sampling_strategy': 'positive_sampling'
        }
        
        sampled_data.append(sampled_entry)
    
    return sampled_data

def create_random_pairs(pocket1_pool, pocket2_pool, num_samples, seed=42, strategy='cross'):
    """
    创建随机配对的pocket pairs
    
    Args:
        pocket1_pool: pocket1池子
        pocket2_pool: pocket2池子  
        num_samples: 要生成的样本数量
        seed: 随机种子
        strategy: 配对策略
            - 'mixed': 混合策略，pocket1和pocket2都可能来自任一池子
            - 'cross': 交叉策略，pocket1来自原pocket1池，pocket2来自原pocket2池，但重新配对
            - 'swap': 交换策略，pocket1来自原pocket2池，pocket2来自原pocket1池
            
    Returns:
        list: 新的数据条目列表
    """
    print(f"Creating {num_samples} random pairs using strategy: {strategy}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    new_data_list = []
    
    for i in tqdm(range(num_samples), desc="Creating random pairs"):

        if strategy == 'cross':
            # 交叉策略：pocket1从原pocket1池选，pocket2从原pocket2池选，但重新配对
            pocket1_info = random.choice(pocket1_pool)
            pocket2_info = random.choice(pocket2_pool)
            
            # 确保不是原来的配对（如果可能的话）
            max_attempts = 100
            attempt = 0
            while attempt < max_attempts:
                if pocket1_info['name'] != pocket2_info['name'].replace('_pocket', '_filtered_peptide'):
                    break
                pocket2_info = random.choice(pocket2_pool)
                attempt += 1
                
            if pocket1_info['name'] == pocket2_info['name'].replace('_pocket', '_filtered_peptide'):
                print(f"Warning: Unable to avoid original pairing for {pocket1_info['name']} after {max_attempts} attempts")
                exit(0)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 创建新的数据条目 - 保留原始pocket名字
        new_data_entry = {
            'pocket1': pocket1_info['name'],  # 保留原始pocket1名字
            'pocket1_atoms': pocket1_info['atoms'].copy(),
            'pocket1_coordinates': pocket1_info['coordinates'].copy(),
            'pocket2': pocket2_info['name'],  # 保留原始pocket2名字
            'pocket2_atoms': pocket2_info['atoms'].copy(),
            'pocket2_coordinates': pocket2_info['coordinates'].copy(),
            'label': 0,  # 负样本标签
            'pairing_index': i,  # 添加配对索引用于标识
            'pairing_strategy': strategy
        }
        
        new_data_list.append(new_data_entry)
    
    return new_data_list

def save_to_lmdb(data_list, output_lmdb):
    """
    将数据保存到LMDB数据库
    
    Args:
        data_list: 数据条目列表
        output_lmdb: 输出LMDB路径
    """
    print(f"Saving {len(data_list)} entries to {output_lmdb}...")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_lmdb), exist_ok=True)
    
    # 创建LMDB环境
    env = lmdb.open(output_lmdb, 
                    map_size=int(1e12),
                    subdir=False,
                    readonly=False)
    
    with env.begin(write=True) as txn:
        for i, data_entry in enumerate(tqdm(data_list, desc="Saving to LMDB")):
            serialized_data = pickle.dumps(data_entry)
            key = str(i).encode('ascii')
            txn.put(key, serialized_data)
    
    env.close()
    print(f"Successfully saved {len(data_list)} entries to {output_lmdb}")

def analyze_data_stats(data_list, title="Data Statistics"):
    """
    分析数据统计信息
    
    Args:
        data_list: 数据条目列表
        title: 统计标题
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # 基本统计
    total_samples = len(data_list)
    label_counts = defaultdict(int)
    
    pocket1_atom_counts = []
    pocket2_atom_counts = []
    
    for entry in data_list:
        label_counts[entry['label']] += 1
        pocket1_atom_counts.append(len(entry['pocket1_atoms']))
        pocket2_atom_counts.append(len(entry['pocket2_atoms']))
    
    print(f"Total samples: {total_samples}")
    print(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  Label {label}: {count} ({count/total_samples*100:.2f}%)")
    
    print(f"\nPocket1 atom statistics:")
    print(f"  Min atoms: {min(pocket1_atom_counts)}")
    print(f"  Max atoms: {max(pocket1_atom_counts)}")
    print(f"  Mean atoms: {np.mean(pocket1_atom_counts):.2f}")
    print(f"  Std atoms: {np.std(pocket1_atom_counts):.2f}")
    
    print(f"\nPocket2 atom statistics:")
    print(f"  Min atoms: {min(pocket2_atom_counts)}")
    print(f"  Max atoms: {max(pocket2_atom_counts)}")
    print(f"  Mean atoms: {np.mean(pocket2_atom_counts):.2f}")
    print(f"  Std atoms: {np.std(pocket2_atom_counts):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Create random pocket pairings for negative samples or sample positive data")
    parser.add_argument("--input_lmdb", required=True, help="Input LMDB file path")
    parser.add_argument("--output_lmdb", required=True, help="Output LMDB file path")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--strategy", choices=['cross', 'positive_sampling'], default='cross',
                       help="Strategy: cross (for negative samples) or positive_sampling (for positive samples)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze input data without generating new samples")
    
    args = parser.parse_args()

    
    # 读取原始数据
    original_data = read_lmdb_data(args.input_lmdb)
    
    # 分析原始数据
    analyze_data_stats(original_data, "Original Data Statistics")
    
    if args.analyze_only:
        print("Analysis complete. Exiting...")
        return
    
    # 根据策略生成样本
    if args.strategy == 'positive_sampling':
        # 正样本抽样
        generated_data = create_positive_samples(
            original_data,
            args.num_samples,
            args.seed
        )
        data_type = "Sampled Positive Samples"
    else:
        # 负样本生成（随机配对）
        # 提取pocket池子
        pocket1_pool, pocket2_pool = extract_pocket_pools(original_data)
        
        # 创建随机配对
        generated_data = create_random_pairs(
            pocket1_pool, 
            pocket2_pool, 
            args.num_samples,
            args.seed,
            args.strategy
        )
        data_type = "Generated Negative Samples"
    
    # 分析生成的数据
    analyze_data_stats(generated_data, f"{data_type} Statistics")
    
    # 保存到LMDB
    save_to_lmdb(generated_data, args.output_lmdb)
    
    print(f"\nSuccessfully created {args.output_lmdb} with {len(generated_data)} samples")
    print(f"Strategy used: {args.strategy}")

if __name__ == "__main__":
    main()
