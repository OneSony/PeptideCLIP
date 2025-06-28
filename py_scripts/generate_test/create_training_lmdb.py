import os
import lmdb
import pickle
import numpy as np
from Bio import PDB
import argparse
from tqdm import tqdm
import random

# 输入的pdb是两条链，作为正匹配

# 原子类型映射（基于UniMol的原子字典）
ATOM_TYPES = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
}

def parse_pdb_chain(pdb_file, chain_id):
    """从PDB文件中解析指定链的原子信息"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    atoms = []
    coordinates = []
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    for atom in residue:
                        if atom.element in ATOM_TYPES:
                            atoms.append(atom.name)  # 改为存储原子名称而不是元素符号
                            coordinates.append(atom.coord)
    
    return atoms, np.array(coordinates)  # atoms 不需要转为 np.array

def get_all_chains_info(pdb_file):
    """获取PDB文件中所有链的信息"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chains_info = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            atom_count = 0
            
            for residue in chain:
                for atom in residue:
                    if atom.element in ATOM_TYPES:
                        atom_count += 1
            
            if atom_count > 0:  # 只记录有效原子数大于0的链
                chains_info[chain_id] = atom_count
    
    return chains_info

def process_pdb_file(pdb_file):
    """处理单个PDB文件，自动根据链长度分配pocket1和pocket2"""
    try:
        # 获取所有链的信息
        chains_info = get_all_chains_info(pdb_file)
        
        # 检查是否有至少两条链
        if len(chains_info) < 2:
            print(f"Warning: Less than 2 chains found in {pdb_file}")
            return None
        
        # 按原子数量排序，取前两条链
        sorted_chains = sorted(chains_info.items(), key=lambda x: x[1])
        short_chain_id, short_chain_length = sorted_chains[0]  # 短链作为pocket1
        long_chain_id, long_chain_length = sorted_chains[-1]   # 长链作为pocket2
        
        print(f"Processing {os.path.basename(pdb_file)}: "
              f"pocket1 (chain {short_chain_id}, {short_chain_length} atoms), "
              f"pocket2 (chain {long_chain_id}, {long_chain_length} atoms)")
        
        # 解析两个链
        pocket1_atoms, pocket1_coords = parse_pdb_chain(pdb_file, short_chain_id)
        pocket2_atoms, pocket2_coords = parse_pdb_chain(pdb_file, long_chain_id)
        
        # 检查是否成功提取到原子
        if len(pocket1_atoms) == 0 or len(pocket2_atoms) == 0:
            print(f"Warning: Empty chain found in {pdb_file}")
            return None
        
        # 构建数据条目
        data_entry = {
            'pocket1': f"{os.path.basename(pdb_file)}_{short_chain_id}",
            'pocket1_atoms': pocket1_atoms,
            'pocket1_coordinates': pocket1_coords.astype(np.float32),
            'pocket2': f"{os.path.basename(pdb_file)}_{long_chain_id}",
            'pocket2_atoms': pocket2_atoms,
            'pocket2_coordinates': pocket2_coords.astype(np.float32),
            'label': 1  # 同一个PDB的两个链，认为是相似的
        }
        
        return data_entry
    
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

def create_lmdb_from_file_list(file_list, pdb_dir, output_lmdb):
    """从文件列表创建LMDB"""
    env = lmdb.open(output_lmdb, 
                map_size=int(1e12),
                subdir=False,      # 重要：创建单文件
                readonly=False)
    
    with env.begin(write=True) as txn:
        valid_count = 0
        for pdb_file in tqdm(file_list, desc=f"Creating {os.path.basename(output_lmdb)}"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            
            data_entry = process_pdb_file(pdb_path)
            
            if data_entry is not None:
                serialized_data = pickle.dumps(data_entry)
                key = str(valid_count).encode('ascii')
                txn.put(key, serialized_data)
                valid_count += 1
    
    env.close()
    print(f"Created {os.path.basename(output_lmdb)} with {valid_count} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to LMDB format for PocketCLIP")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LMDB files")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio for training set (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # 设置随机种子以确保可重现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")
    
    # 随机打乱文件列表
    random.shuffle(pdb_files)
    
    # 划分训练集和验证集
    split_idx = int(len(pdb_files) * args.train_ratio)
    train_files = pdb_files[:split_idx]
    valid_files = pdb_files[split_idx:]
    
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(valid_files)} files")
    
    # 创建训练集LMDB
    train_lmdb_path = os.path.join(args.output_dir, "train.lmdb")
    create_lmdb_from_file_list(train_files, args.pdb_dir, train_lmdb_path)
    
    # 创建验证集LMDB
    valid_lmdb_path = os.path.join(args.output_dir, "valid.lmdb")
    create_lmdb_from_file_list(valid_files, args.pdb_dir, valid_lmdb_path)
    
    print("Conversion completed!")
    print(f"Training set saved to: {train_lmdb_path}")
    print(f"Validation set saved to: {valid_lmdb_path}")