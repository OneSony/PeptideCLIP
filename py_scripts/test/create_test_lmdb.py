import os
import lmdb
import pickle
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO
import argparse
from tqdm import tqdm
from scipy.spatial.distance import cdist
import random

# 输入的pdb是两条链，peptide和receptor，提取peptide和其周围的pocket

# 原子类型映射（基于UniMol的原子字典）
ATOM_TYPES = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
}

class PeptideSelector:
    """选择peptide链的原子"""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_model(self, model):
        return True
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
    def accept_residue(self, residue):
        return True
    
    def accept_atom(self, atom):
        return True

class PocketSelector:
    """选择pocket残基的原子"""
    def __init__(self, chain_id, residue_ids):
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)
    
    def accept_model(self, model):
        return True
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
    def accept_residue(self, residue):
        return residue.id in self.residue_ids
    
    def accept_atom(self, atom):
        return True

def determine_peptide_receptor_chains(pdb_file):
    """自动判断哪个链是peptide，哪个是receptor（基于链长度）"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chain_info = []
    
    for model in structure:
        for chain in model:
            residue_count = len([res for res in chain if res.id[0] == ' '])
            if residue_count > 0:  # 只考虑有残基的链
                chain_info.append((chain.id, residue_count))
    
    if len(chain_info) < 2:
        raise ValueError(f"PDB file {pdb_file} does not contain at least 2 chains")
    
    # 按残基数量排序
    chain_info.sort(key=lambda x: x[1])
    
    # 最短的是peptide，最长的是receptor
    peptide_chain = chain_info[0][0]
    receptor_chain = chain_info[-1][0]
    
    #print(f"Detected chains in {os.path.basename(pdb_file)}: "
    #      f"peptide={peptide_chain}({chain_info[0][1]} residues), "
    #      f"receptor={receptor_chain}({chain_info[-1][1]} residues)")
    
    return peptide_chain, receptor_chain

def save_pocket_pdb(pdb_file, receptor_chain, pocket_residue_ids, output_dir):
    """保存pocket残基为单独的PDB文件"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    io = PDBIO()
    io.set_structure(structure)
    
    output_filename = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{receptor_chain}_pocket.pdb"
    output_path = os.path.join(output_dir, output_filename)
    
    io.save(output_path, PocketSelector(receptor_chain, pocket_residue_ids))
    return output_path

def extract_interacting_regions(peptide_atoms, peptide_coords, peptide_residues,
                               receptor_atoms, receptor_coords, receptor_residues, 
                               cutoff=5.0):
    """双向提取相互作用的peptide残基和pocket残基"""
    if len(peptide_coords) == 0 or len(receptor_coords) == 0:
        return [], np.array([]), [], [], np.array([]), []
    
    # 计算peptide和receptor原子之间的距离矩阵
    distances = cdist(peptide_coords, receptor_coords)
    
    # 找到距离小于cutoff的原子对
    peptide_indices, receptor_indices = np.where(distances < cutoff)
    
    # 获取相互作用的peptide残基ID
    interacting_peptide_residue_ids = set()
    for idx in peptide_indices:
        for res_info in peptide_residues:
            if res_info['atom_index'] == idx:
                interacting_peptide_residue_ids.add(res_info['residue_id'])
                break
    
    # 获取相互作用的receptor残基ID（pocket）
    interacting_receptor_residue_ids = set()
    for idx in receptor_indices:
        for res_info in receptor_residues:
            if res_info['atom_index'] == idx:
                interacting_receptor_residue_ids.add(res_info['residue_id'])
                break
    
    # 提取相互作用的peptide残基的所有原子
    filtered_peptide_atoms = []
    filtered_peptide_coords = []
    
    for i, res_info in enumerate(peptide_residues):
        if res_info['residue_id'] in interacting_peptide_residue_ids:
            filtered_peptide_atoms.append(peptide_atoms[res_info['atom_index']])
            filtered_peptide_coords.append(peptide_coords[res_info['atom_index']])
    
    # 提取相互作用的receptor残基的所有原子（pocket）
    pocket_atoms = []
    pocket_coords = []
    
    for i, res_info in enumerate(receptor_residues):
        if res_info['residue_id'] in interacting_receptor_residue_ids:
            pocket_atoms.append(receptor_atoms[res_info['atom_index']])
            pocket_coords.append(receptor_coords[res_info['atom_index']])
    
    return (filtered_peptide_atoms, np.array(filtered_peptide_coords), list(interacting_peptide_residue_ids),
            pocket_atoms, np.array(pocket_coords), list(interacting_receptor_residue_ids))

def parse_chain_atoms_with_residue_info(pdb_file, chain_id):
    """从PDB文件中解析指定链的原子信息，包含完整的残基信息"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    atoms = []
    coordinates = []
    residues_info = []
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    for atom in residue:
                        if atom.element in ATOM_TYPES:
                            atoms.append(atom.name)
                            coordinates.append(atom.coord)
                            residues_info.append({
                                'residue_id': residue.id,
                                'residue_name': residue.resname,
                                'atom_index': len(atoms) - 1
                            })
    
    return atoms, np.array(coordinates), residues_info

class FilteredPeptideSelector:
    """选择经过距离筛选的peptide残基"""
    def __init__(self, chain_id, residue_ids):
        self.chain_id = chain_id
        self.residue_ids = set(residue_ids)
    
    def accept_model(self, model):
        return True
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    
    def accept_residue(self, residue):
        return residue.id in self.residue_ids
    
    def accept_atom(self, atom):
        return True

def save_filtered_peptide_pdb(pdb_file, peptide_chain, peptide_residue_ids, output_dir):
    """保存经过筛选的peptide残基为PDB文件"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    io = PDBIO()
    io.set_structure(structure)
    
    output_filename = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{peptide_chain}_filtered_peptide.pdb"
    output_path = os.path.join(output_dir, output_filename)
    
    io.save(output_path, FilteredPeptideSelector(peptide_chain, peptide_residue_ids))
    return output_path

def save_filtered_complex_pdb(pdb_file, peptide_chain, peptide_residue_ids, 
                             receptor_chain, pocket_residue_ids, output_dir):
    """保存经过筛选的peptide和pocket合并的复合体PDB文件"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 创建一个新的结构用于保存复合体
    new_structure = PDB.Structure.Structure('complex')
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)
    
    # 添加筛选后的peptide残基
    new_peptide_chain = PDB.Chain.Chain('P')  # P链表示peptide
    peptide_residue_ids_set = set(peptide_residue_ids)
    
    for model in structure:
        for chain in model:
            if chain.id == peptide_chain:
                for residue in chain:
                    if residue.id in peptide_residue_ids_set:
                        new_residue = residue.copy()
                        new_peptide_chain.add(new_residue)
    
    new_model.add(new_peptide_chain)
    
    # 添加pocket残基作为新链
    new_pocket_chain = PDB.Chain.Chain('K')  # K链表示pocket
    pocket_residue_ids_set = set(pocket_residue_ids)
    
    for model in structure:
        for chain in model:
            if chain.id == receptor_chain:
                for residue in chain:
                    if residue.id in pocket_residue_ids_set:
                        new_residue = residue.copy()
                        new_pocket_chain.add(new_residue)
    
    new_model.add(new_pocket_chain)
    
    # 保存复合体PDB文件
    io = PDBIO()
    io.set_structure(new_structure)
    
    output_filename = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_filtered_complex.pdb"
    output_path = os.path.join(output_dir, output_filename)
    
    io.save(output_path)
    return output_path

def process_pdb_file(pdb_file, cutoff=5.0, peptide_dir=None, pocket_dir=None, complex_dir=None, label=1):
    """处理单个PDB文件，提取peptide和对应的pocket，进行双向筛选"""
    try:
        # 自动判断哪个链是peptide，哪个是receptor
        peptide_chain, receptor_chain = determine_peptide_receptor_chains(pdb_file)
        
        # 解析peptide链
        peptide_atoms, peptide_coords, peptide_residues = parse_chain_atoms_with_residue_info(
            pdb_file, peptide_chain)
        
        # 解析receptor链
        receptor_atoms, receptor_coords, receptor_residues = parse_chain_atoms_with_residue_info(
            pdb_file, receptor_chain)
        
        # 检查是否成功提取到原子
        if len(peptide_atoms) == 0 or len(receptor_atoms) == 0:
            print(f"Warning: Empty chain found in {pdb_file}")
            return None
        
        # 双向提取相互作用的区域
        (filtered_peptide_atoms, filtered_peptide_coords, peptide_residue_ids,
         pocket_atoms, pocket_coords, pocket_residue_ids) = extract_interacting_regions(
            peptide_atoms, peptide_coords, peptide_residues,
            receptor_atoms, receptor_coords, receptor_residues, cutoff)
        
        if len(filtered_peptide_atoms) == 0 or len(pocket_atoms) == 0:
            print(f"Warning: No interacting residues found for {pdb_file}")
            return None
        
        # 保存筛选后的peptide和pocket为单独的PDB文件
        peptide_pdb_path = None
        pocket_pdb_path = None
        complex_pdb_path = None
        
        if peptide_dir is not None:
            peptide_pdb_path = save_filtered_peptide_pdb(
                pdb_file, peptide_chain, peptide_residue_ids, peptide_dir)
        
        if pocket_dir is not None:
            pocket_pdb_path = save_pocket_pdb(
                pdb_file, receptor_chain, pocket_residue_ids, pocket_dir)
        
        if complex_dir is not None:
            complex_pdb_path = save_filtered_complex_pdb(
                pdb_file, peptide_chain, peptide_residue_ids, 
                receptor_chain, pocket_residue_ids, complex_dir)
        
        print(f"Processed {os.path.basename(pdb_file)}: "
              f"peptide residues: {len(peptide_residue_ids)}, "
              f"pocket residues: {len(pocket_residue_ids)}")
        
        # 构建数据条目
        data_entry = {
            'pocket1': f"{os.path.basename(pdb_file)}_{peptide_chain}_filtered_peptide",
            'pocket1_atoms': filtered_peptide_atoms,
            'pocket1_coordinates': filtered_peptide_coords.astype(np.float32),
            'pocket2': f"{os.path.basename(pdb_file)}_{receptor_chain}_pocket",
            'pocket2_atoms': pocket_atoms,
            'pocket2_coordinates': pocket_coords.astype(np.float32),
            'label': label,  # 使用传入的label参数
        }
        
        return data_entry
    
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

def create_lmdb_from_file_list(file_list, pdb_dir, output_lmdb, cutoff, positive_files_set=None, peptide_dir=None, pocket_dir=None, complex_dir=None):
    """从文件列表创建LMDB"""
    # 创建输出目录
    if peptide_dir is not None:
        os.makedirs(peptide_dir, exist_ok=True)
    else:
        peptide_dir = None
        
    if pocket_dir is not None:
        os.makedirs(pocket_dir, exist_ok=True)
    else:
        pocket_dir = None
    
    if complex_dir is not None:
        os.makedirs(complex_dir, exist_ok=True)
    else:
        complex_dir = None
    
    env = lmdb.open(output_lmdb, 
                map_size=int(1e12),
                subdir=False,      # 重要：创建单文件
                readonly=False)
    
    with env.begin(write=True) as txn:
        valid_count = 0
        positive_count = 0
        negative_count = 0
        
        for pdb_file in tqdm(file_list, desc=f"Creating {os.path.basename(output_lmdb)}"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            
            # 根据文件名是否在positive_files_set中来确定label
            if positive_files_set is not None:
                label = 1 if pdb_file in positive_files_set else 0
            else:
                label = 1  # 如果没有提供positive文件列表，默认都是1
            
            data_entry = process_pdb_file(pdb_path, cutoff, peptide_dir, pocket_dir, complex_dir, label)
            
            if data_entry is not None:
                serialized_data = pickle.dumps(data_entry)
                key = str(valid_count).encode('ascii')
                txn.put(key, serialized_data)
                valid_count += 1
                
                if label == 1:
                    positive_count += 1
                else:
                    negative_count += 1
    
    env.close()
    print(f"Created {os.path.basename(output_lmdb)} with {valid_count} entries")
    print(f"  - Positive samples (label=1): {positive_count}")
    print(f"  - Negative samples (label=0): {negative_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to LMDB format for PeptideCLIP")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for LMDB files")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for pocket extraction (Angstroms)")
    parser.add_argument("--positive_list", type=str, default=None, help="Text file containing PDB filenames with label=1 (one per line)")
    parser.add_argument("--peptide_dir", type=str, default=None, help="Directory to save peptide PDB files")
    parser.add_argument("--pocket_dir", type=str, default=None, help="Directory to save pocket PDB files")
    parser.add_argument("--complex_dir", type=str, default=None, help="Directory to save complex PDB files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting")
    
    args = parser.parse_args()
    
    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")
    
    # 读取positive文件列表
    positive_files_set = None
    if args.positive_list is not None:
        if os.path.exists(args.positive_list):
            with open(args.positive_list, 'r') as f:
                positive_files_set = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(positive_files_set)} positive files from {args.positive_list}")
        else:
            print(f"Warning: Positive list file {args.positive_list} not found. All files will have label=1")

    # 定义输出文件路径
    test_lmdb = args.output_path
    
    # 创建测试集
    create_lmdb_from_file_list(pdb_files, args.pdb_dir, test_lmdb, 
                             args.cutoff, positive_files_set, args.peptide_dir, args.pocket_dir, args.complex_dir)

    print("Conversion completed!")
    