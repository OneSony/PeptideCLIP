import os
import lmdb
import pickle
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO
import argparse
from tqdm import tqdm
from scipy.spatial.distance import cdist

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

def save_filtered_pocket_pdb(pdb_file, receptor_chain, pocket_residue_ids, output_dir):
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

def process_pdb_file(pdb_file, cutoff=5.0, peptide_dir=None, pocket_dir=None, complex_dir=None, positive_files=None, receptor_lmdb_dir=None):
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
            pocket_pdb_path = save_filtered_pocket_pdb(
                pdb_file, receptor_chain, pocket_residue_ids, pocket_dir)
        
        if complex_dir is not None:
            complex_pdb_path = save_filtered_complex_pdb(
                pdb_file, peptide_chain, peptide_residue_ids, 
                receptor_chain, pocket_residue_ids, complex_dir)
        
        print(f"Processed {os.path.basename(pdb_file)}: "
              f"peptide residues: {len(peptide_residue_ids)}, "
              f"pocket residues: {len(pocket_residue_ids)}")
        
        # 确定标签：检查文件是否在正面列表中
        base_filename = os.path.splitext(os.path.basename(pdb_file))[0]
        if positive_files is not None:
            label = 1 if base_filename in positive_files else 0
        else:
            label = 1  # 如果没有提供列表，默认为1
        
        # 如果是positive文件且指定了receptor_lmdb_dir，为receptor创建独立的LMDB
        if receptor_lmdb_dir is not None and positive_files is not None and base_filename in positive_files:
            receptor_name = f"{base_filename}_{receptor_chain}_receptor"
            create_receptor_lmdb(pdb_file, pocket_atoms, pocket_coords, receptor_name, receptor_lmdb_dir)
        
        # 构建数据条目 - 只保存peptide信息
        data_entry = {
            'pocket_atoms': filtered_peptide_atoms,  # 保持原子名称
            'pocket_coordinates': filtered_peptide_coords.astype(np.float32),
            'pocket': f"{os.path.basename(pdb_file)}_{peptide_chain}",
            'label': label,  # 根据文件列表设置标签
        }
        
        return data_entry
    
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None

def create_lmdb_from_file_list(file_list, pdb_dir, output_lmdb, cutoff, peptide_dir=None, pocket_dir=None, complex_dir=None, dataset_type="data", positive_files=None, receptor_lmdb_dir=None):
    """从文件列表创建LMDB"""
    # 创建输出目录
    if peptide_dir is not None:
        peptide_output_dir = os.path.join(peptide_dir, dataset_type)
        os.makedirs(peptide_output_dir, exist_ok=True)
    else:
        peptide_output_dir = None
        
    if pocket_dir is not None:
        pocket_output_dir = os.path.join(pocket_dir, dataset_type)
        os.makedirs(pocket_output_dir, exist_ok=True)
    else:
        pocket_output_dir = None
    
    if complex_dir is not None:
        complex_output_dir = os.path.join(complex_dir, dataset_type)
        os.makedirs(complex_output_dir, exist_ok=True)
    else:
        complex_output_dir = None
    
    env = lmdb.open(output_lmdb, 
                map_size=int(1e12),
                subdir=False,      # 重要：创建单文件
                readonly=False)
    
    with env.begin(write=True) as txn:
        valid_count = 0
        for pdb_file in tqdm(file_list, desc=f"Creating {dataset_type} {os.path.basename(output_lmdb)}"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            
            data_entry = process_pdb_file(pdb_path, cutoff, peptide_output_dir, pocket_output_dir, complex_output_dir, positive_files, receptor_lmdb_dir)
            
            if data_entry is not None:
                serialized_data = pickle.dumps(data_entry)
                key = str(valid_count).encode('ascii')
                txn.put(key, serialized_data)
                valid_count += 1
    
    env.close()
    print(f"Created {dataset_type} {os.path.basename(output_lmdb)} with {valid_count} entries")

def load_positive_list(list_file):
    """加载标签为1的文件名列表"""
    if list_file is None:
        return set()
    
    positive_files = set()
    with open(list_file, 'r') as f:
        for line in f:
            filename = line.strip()
            if filename:
                # 去除可能的.pdb后缀，确保统一格式
                if filename.endswith('.pdb'):
                    filename = filename[:-4]
                positive_files.add(filename)
    
    print(f"Loaded {len(positive_files)} positive files from {list_file}")
    return positive_files

def create_receptor_lmdb(pdb_file, receptor_atoms, receptor_coords, receptor_name, output_dir):
    """为单个receptor创建独立的LMDB文件"""
    
    # 构建receptor数据条目 - 使用与peptide相同的逻辑
    data_entry = {
        'pocket_atoms': receptor_atoms,  # 保持原子名称
        'pocket_coordinates': receptor_coords.astype(np.float32),
        'pocket': receptor_name,
    }
    
    # 创建LMDB文件
    base_filename = os.path.splitext(os.path.basename(pdb_file))[0]
    lmdb_filename = f"{base_filename}.lmdb"
    lmdb_path = os.path.join(output_dir, lmdb_filename)
    
    env = lmdb.open(lmdb_path, 
                map_size=int(1e9),  # 单个receptor较小，使用1GB
                subdir=False,
                readonly=False)
    
    with env.begin(write=True) as txn:
        serialized_data = pickle.dumps(data_entry)
        key = str(0).encode('ascii')  # 单个条目，使用key=0
        txn.put(key, serialized_data)
    
    env.close()
    print(f"Created receptor LMDB: {lmdb_path}")
    return lmdb_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDB files to LMDB format for PeptideCLIP")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files")
    parser.add_argument("--output_lmdb", type=str, required=True, help="Output LMDB file path")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff for pocket extraction (Angstroms)")
    parser.add_argument("--peptide_dir", type=str, default=None, help="Directory to save peptide PDB files")
    parser.add_argument("--receptor_dir", type=str, default=None, help="Directory to save receptor PDB files")
    parser.add_argument("--receptor_lmdb_dir", type=str, default=None, help="Directory to save individual receptor LMDB files")
    parser.add_argument("--positive_list", type=str, default=None, help="File containing list of positive PDB files (label=1)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_lmdb)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建receptor LMDB输出目录
    if args.receptor_lmdb_dir:
        os.makedirs(args.receptor_lmdb_dir, exist_ok=True)
    
    # 加载正面文件列表
    positive_files = load_positive_list(args.positive_list)
    
    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")
    
    # 创建LMDB
    create_lmdb_from_file_list(pdb_files, args.pdb_dir, args.output_lmdb, 
                             args.cutoff, args.peptide_dir, args.receptor_dir, None, "data", positive_files, args.receptor_lmdb_dir)
    
    print("Conversion completed!")
    print(f"Created LMDB: {args.output_lmdb}")
    
    # 统计标签分布
    if positive_files:
        processed_positive = sum(1 for f in pdb_files if os.path.splitext(f)[0] in positive_files)
        processed_negative = len(pdb_files) - processed_positive
        print(f"Label distribution: {processed_positive} positive (label=1), {processed_negative} negative (label=0)")