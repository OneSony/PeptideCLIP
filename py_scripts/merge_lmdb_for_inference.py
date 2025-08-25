#!/usr/bin/env python3
"""
Merge two LMDB files with different labels for PeptideCLIP inference

This script takes two LMDB files as input, assigns different labels to each,
and creates a single output LMDB file suitable for inference.

Usage:
    python merge_lmdb_for_inference.py --lmdb1 path/to/first.lmdb --lmdb2 path/to/second.lmdb \
                                      --output path/to/output.lmdb \
                                      --label1 1 --label2 0
"""

import os
import lmdb
import pickle
import argparse
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_lmdb_entries(lmdb_path):
    """读取LMDB文件中的所有条目"""
    entries = []
    
    env = lmdb.open(lmdb_path, 
                   subdir=False,
                   readonly=True,
                   lock=False,
                   readahead=False,
                   meminit=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data_entry = pickle.loads(value)
                entries.append(data_entry)
            except Exception as e:
                logger.warning(f"Failed to load entry with key {key}: {e}")
                continue
    
    env.close()
    logger.info(f"Read {len(entries)} entries from {os.path.basename(lmdb_path)}")
    return entries


def validate_data_entry(data_entry):
    """验证数据条目是否包含必需的字段"""
    required_fields = [
        'pocket1', 'pocket1_atoms', 'pocket1_coordinates',
        'pocket2', 'pocket2_atoms', 'pocket2_coordinates'
    ]
    
    for field in required_fields:
        if field not in data_entry:
            return False, f"Missing field: {field}"
    
    # 检查数据类型和形状
    if len(data_entry['pocket1_atoms']) != len(data_entry['pocket1_coordinates']):
        return False, "pocket1_atoms and pocket1_coordinates length mismatch"
    
    if len(data_entry['pocket2_atoms']) != len(data_entry['pocket2_coordinates']):
        return False, "pocket2_atoms and pocket2_coordinates length mismatch"
    
    return True, "Valid"


def merge_lmdb_files(lmdb1_path, lmdb2_path, output_path, label1=1, label2=0, shuffle=True, sample_lmdb1=False):
    """
    合并两个LMDB文件并分配不同的标签
    
    Args:
        lmdb1_path: 第一个LMDB文件路径
        lmdb2_path: 第二个LMDB文件路径
        output_path: 输出LMDB文件路径
        label1: 分配给第一个LMDB的标签
        label2: 分配给第二个LMDB的标签
        shuffle: 是否打乱合并后的数据顺序
        sample_lmdb1: 是否从LMDB1中随机抽取和LMDB2相同数量的样本
    """
    
    # 读取两个LMDB文件
    logger.info(f"Reading first LMDB: {lmdb1_path}")
    entries1 = read_lmdb_entries(lmdb1_path)
    
    logger.info(f"Reading second LMDB: {lmdb2_path}")
    entries2 = read_lmdb_entries(lmdb2_path)
    
    # 如果需要，从LMDB1中随机抽取样本
    if sample_lmdb1:
        num_lmdb2 = len(entries2)
        if len(entries1) > num_lmdb2:
            import random
            logger.info(f"Randomly sampling {num_lmdb2} entries from LMDB1 (original: {len(entries1)})")
            entries1 = random.sample(entries1, num_lmdb2)
        elif len(entries1) < num_lmdb2:
            logger.warning(f"LMDB1 has fewer entries ({len(entries1)}) than LMDB2 ({num_lmdb2}). "
                          f"Using all entries from LMDB1.")
        else:
            logger.info(f"LMDB1 and LMDB2 have the same number of entries ({len(entries1)})")
    
    # 为条目分配标签
    logger.info(f"Assigning label {label1} to entries from first LMDB")
    for entry in entries1:
        entry['label'] = label1
    
    logger.info(f"Assigning label {label2} to entries from second LMDB")
    for entry in entries2:
        entry['label'] = label2
    
    # 合并所有条目
    all_entries = entries1 + entries2
    logger.info(f"Total entries: {len(all_entries)} "
                f"(Label {label1}: {len(entries1)}, Label {label2}: {len(entries2)})")
    
    # 打乱顺序（如果需要）
    if shuffle:
        import random
        random.shuffle(all_entries)
        logger.info("Shuffled the merged entries")
    
    # 验证数据条目
    valid_entries = []
    invalid_count = 0
    
    logger.info("Validating data entries...")
    for i, entry in enumerate(tqdm(all_entries, desc="Validating")):
        is_valid, message = validate_data_entry(entry)
        if is_valid:
            valid_entries.append(entry)
        else:
            invalid_count += 1
            logger.warning(f"Invalid entry {i}: {message}")
    
    logger.info(f"Valid entries: {len(valid_entries)}, Invalid entries: {invalid_count}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入输出LMDB
    logger.info(f"Writing merged LMDB to: {output_path}")
    env = lmdb.open(output_path,
                   map_size=int(1e12),
                   subdir=False,
                   readonly=False)
    
    with env.begin(write=True) as txn:
        for i, entry in enumerate(tqdm(valid_entries, desc="Writing LMDB")):
            try:
                serialized_data = pickle.dumps(entry)
                key = str(i).encode('ascii')
                txn.put(key, serialized_data)
            except Exception as e:
                logger.error(f"Failed to write entry {i}: {e}")
                continue
    
    env.close()
    logger.info(f"Successfully created merged LMDB with {len(valid_entries)} entries")
    
    return len(valid_entries)


def print_lmdb_stats(lmdb_path, max_samples=5):
    """打印LMDB文件的统计信息"""
    logger.info(f"Statistics for {os.path.basename(lmdb_path)}:")
    
    entries = read_lmdb_entries(lmdb_path)
    
    if len(entries) == 0:
        logger.info("  No entries found")
        return
    
    # 统计标签分布
    label_counts = {}
    pocket1_atom_counts = []
    pocket2_atom_counts = []
    
    for entry in entries:
        label = entry.get('label', 'Unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
        
        pocket1_atom_counts.append(len(entry.get('pocket1_atoms', [])))
        pocket2_atom_counts.append(len(entry.get('pocket2_atoms', [])))
    
    logger.info(f"  Total entries: {len(entries)}")
    logger.info(f"  Label distribution: {label_counts}")
    
    if pocket1_atom_counts:
        logger.info(f"  Pocket1 atoms - Min: {min(pocket1_atom_counts)}, "
                   f"Max: {max(pocket1_atom_counts)}, "
                   f"Avg: {sum(pocket1_atom_counts)/len(pocket1_atom_counts):.1f}")
    
    if pocket2_atom_counts:
        logger.info(f"  Pocket2 atoms - Min: {min(pocket2_atom_counts)}, "
                   f"Max: {max(pocket2_atom_counts)}, "
                   f"Avg: {sum(pocket2_atom_counts)/len(pocket2_atom_counts):.1f}")
    
    # 显示前几个样本的详细信息
    logger.info(f"  Sample entries (first {min(max_samples, len(entries))}):")
    for i in range(min(max_samples, len(entries))):
        entry = entries[i]
        logger.info(f"    Entry {i}:")
        logger.info(f"      pocket1: {entry.get('pocket1', 'N/A')}")
        logger.info(f"      pocket2: {entry.get('pocket2', 'N/A')}")
        logger.info(f"      label: {entry.get('label', 'N/A')}")
        logger.info(f"      pocket1_atoms: {len(entry.get('pocket1_atoms', []))}")
        logger.info(f"      pocket2_atoms: {len(entry.get('pocket2_atoms', []))}")


def main():
    parser = argparse.ArgumentParser(description="Merge two LMDB files with different labels for PeptideCLIP inference")
    parser.add_argument("--lmdb1", required=True, help="Path to first LMDB file")
    parser.add_argument("--lmdb2", required=True, help="Path to second LMDB file")
    parser.add_argument("--output", required=True, help="Path to output merged LMDB file")
    parser.add_argument("--label1", type=int, default=1, help="Label for first LMDB (default: 1)")
    parser.add_argument("--label2", type=int, default=0, help="Label for second LMDB (default: 0)")
    parser.add_argument("--no_shuffle", action="store_true", help="Don't shuffle the merged entries")
    parser.add_argument("--sample_lmdb1", action="store_true", help="Randomly sample from LMDB1 to match LMDB2 size")
    parser.add_argument("--stats", action="store_true", help="Print statistics for input and output LMDB files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    
    args = parser.parse_args()
    
    # 设置随机种子
    import random
    random.seed(args.seed)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.lmdb1):
        raise FileNotFoundError(f"First LMDB file not found: {args.lmdb1}")
    
    if not os.path.exists(args.lmdb2):
        raise FileNotFoundError(f"Second LMDB file not found: {args.lmdb2}")
    
    # 打印输入文件统计信息（如果需要）
    if args.stats:
        print_lmdb_stats(args.lmdb1)
        print_lmdb_stats(args.lmdb2)
        print("=" * 50)
    
    # 合并LMDB文件
    logger.info("Starting LMDB merge process...")
    total_entries = merge_lmdb_files(
        lmdb1_path=args.lmdb1,
        lmdb2_path=args.lmdb2,
        output_path=args.output,
        label1=args.label1,
        label2=args.label2,
        shuffle=not args.no_shuffle,
        sample_lmdb1=args.sample_lmdb1
    )
    
    # 打印输出文件统计信息（如果需要）
    if args.stats:
        print("=" * 50)
        print_lmdb_stats(args.output)
    
    logger.info("=" * 50)
    logger.info("Merge completed successfully!")
    logger.info(f"Input files:")
    logger.info(f"  LMDB1 (label {args.label1}): {args.lmdb1}")
    logger.info(f"  LMDB2 (label {args.label2}): {args.lmdb2}")
    logger.info(f"Output file:")
    logger.info(f"  Merged LMDB: {args.output}")
    logger.info(f"  Total entries: {total_entries}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
