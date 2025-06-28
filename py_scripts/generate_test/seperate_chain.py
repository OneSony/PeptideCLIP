import os
from Bio import PDB
from Bio.SeqUtils import seq1

# 读取文件中的信息
with open("bcma") as f:
    lines = f.readlines()

# 输出目录
match_dir = "peptide"
non_match_dir = "receptor"
os.makedirs(match_dir, exist_ok=True)
os.makedirs(non_match_dir, exist_ok=True)

# 解析PDB文件的函数
def process_pdb_file(pdb_file, target_seq):
    # 解析结构
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    # 获取链与其氨基酸序列的映射
    chain_seq_map = {}

    for model in structure:
        for chain in model:
            seq = ""
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    resname = residue.get_resname()
                    try:
                        seq += seq1(resname)
                    except:
                        pass
            chain_seq_map[chain.id] = seq

    # 查找匹配的链
    matched_chain = None
    for cid, seq in chain_seq_map.items():
        if target_seq in seq or seq in target_seq:  # 支持部分匹配或完全包含
            matched_chain = cid
            break

    if matched_chain is None:
        print(f"!!!!!No chain matches the target sequence in {pdb_file}")
        return

    # 保存匹配和非匹配链
    io = PDB.PDBIO()

    def save_chain(structure, chain_id, output_path):
        class ChainSelect(PDB.Select):
            def accept_chain(self, chain):
                return chain.id == chain_id
        io.set_structure(structure)
        io.save(output_path, ChainSelect())

    for cid in chain_seq_map:
        basename = os.path.basename(pdb_file)
        out_path = os.path.join(match_dir if cid == matched_chain else non_match_dir,
                    f"{os.path.splitext(basename)[0]}.pdb")
        print(f"Saving chain {cid} to {out_path}")
        save_chain(structure[0], cid, out_path)

# 循环处理每一行数据
for line in lines:
    line = line.strip()
    pdb_file, target_seq = line.split('\t')
    print(f"Processing {pdb_file} with target sequence: {target_seq}")
    process_pdb_file(pdb_file, target_seq)
