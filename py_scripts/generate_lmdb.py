import os
import lmdb
import pickle
import numpy as np
from Bio import PDB
import argparse
from tqdm import tqdm

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

def process_pdb_file(pdb_file, chain1_id='A', chain2_id='B'):
    """处理单个PDB文件，提取两个链的信息"""
    try:
        # 解析两个链
        pocket1_atoms, pocket1_coords = parse_pdb_chain(pdb_file, chain1_id)
        pocket2_atoms, pocket2_coords = parse_pdb_chain(pdb_file, chain2_id)
        
        # 检查是否成功提取到原子
        if len(pocket1_atoms) == 0 or len(pocket2_atoms) == 0:
            print(f"Warning: Empty chain found in {pdb_file}")
            return None
        
        # 构建数据条目
        data_entry = {
            'pocket1': f"{os.path.basename(pdb_file)}_{chain1_id}",
            'pocket1_atoms': pocket1_atoms,
            'pocket1_coordinates': pocket1_coords.astype(np.float32),
            'pocket2': f"{os.path.basename(pdb_file)}_{chain2_id}",
            'pocket2_atoms': pocket2_atoms,
            'pocket2_coordinates': pocket2_coords.astype(np.float32),
            'label': 1  # 同一个PDB的两个链，认为是相似的
        }
        
        return data_entry
    
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

def create_lmdb_from_file_list(file_list, pdb_dir, output_lmdb, chain1_id, chain2_id):
    """从文件列表创建LMDB"""
    env = lmdb.open(output_lmdb, 
                map_size=int(1e12),
                subdir=False,      # 重要：创建单文件
                readonly=False)
    
    with env.begin(write=True) as txn:
        valid_count = 0
        for pdb_file in tqdm(file_list, desc=f"Creating {os.path.basename(output_lmdb)}"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            
            data_entry = process_pdb_file(pdb_path, chain1_id, chain2_id)
            
            if data_entry is not None:
                serialized_data = pickle.dumps(data_entry)
                key = str(valid_count).encode('ascii')
                txn.put(key, serialized_data)
                valid_count += 1
        
        # 存储实际的数据集大小
        #txn.put('length'.encode('ascii'), pickle.dumps(valid_count))
    
    env.close()
    print(f"Created {os.path.basename(output_lmdb)} with {valid_count} entries")

def create_train_valid_split(pdb_dir, output_dir, train_ratio=0.8, chain1_id='A', chain2_id='B'):
    """创建训练和验证集"""
    
    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")
    
    # 随机打乱并分割数据
    np.random.shuffle(pdb_files)
    split_idx = int(len(pdb_files) * train_ratio)
    train_files = pdb_files[:split_idx]
    valid_files = pdb_files[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建训练集
    train_lmdb = os.path.join(output_dir, 'train.lmdb')
    create_lmdb_from_file_list(train_files, pdb_dir, train_lmdb, chain1_id, chain2_id)
    
    # 创建验证集
    valid_lmdb = os.path.join(output_dir, 'valid.lmdb')
    create_lmdb_from_file_list(valid_files, pdb_dir, valid_lmdb, chain1_id, chain2_id)
    
    print(f"Dataset split completed:")
    print(f"  Train set: {len(train_files)} files")
    print(f"  Valid set: {len(valid_files)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to LMDB format for PocketCLIP")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LMDB files")
    parser.add_argument("--chain1", type=str, default='L', help="First chain ID")
    parser.add_argument("--chain2", type=str, default='R', help="Second chain ID")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)")
    
    args = parser.parse_args()
    
    # 创建训练和验证集
    create_train_valid_split(
        args.pdb_dir, 
        args.output_dir, 
        args.train_ratio, 
        args.chain1, 
        args.chain2
    )
    
    print("Conversion completed!")