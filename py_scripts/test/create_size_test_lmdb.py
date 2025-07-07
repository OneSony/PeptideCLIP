import lmdb
import pickle
import argparse

def overwrite_pocket1_fields(input_lmdb, output_lmdb):
    env_in = lmdb.open(input_lmdb, readonly=True, lock=False, subdir=False)
    env_out = lmdb.open(output_lmdb, map_size=int(1e12), subdir=False)
    
    # 先遍历一遍，找到目标name对应的pocket1内容（只匹配点之前的内容）
    target_entry = None
    with env_in.begin() as txn_in:
        cursor = txn_in.cursor()
        for key, value in cursor:
            entry = pickle.loads(value)
            pocket1_name = entry.get('pocket1', '')
            # 只取第一个点之前的内容
            pocket1_prefix = pocket1_name.split('.', 1)[0]
            if pocket1_prefix == overwrite_pocket1_fields.target_name:
                target_entry = {
                    'pocket1': entry['pocket1'],
                    'pocket1_atoms': entry['pocket1_atoms'],
                    'pocket1_coordinates': entry['pocket1_coordinates'],
                }
                break
    if target_entry is None:
        raise ValueError(f"未找到指定的pocket1 name前缀: {overwrite_pocket1_fields.target_name}")

    # 再遍历一遍，写入新lmdb
    with env_in.begin() as txn_in, env_out.begin(write=True) as txn_out:
        cursor = txn_in.cursor()
        for key, value in cursor:
            entry = pickle.loads(value)
            entry['pocket1'] = target_entry['pocket1']
            entry['pocket1_atoms'] = target_entry['pocket1_atoms']
            entry['pocket1_coordinates'] = target_entry['pocket1_coordinates']
            txn_out.put(key, pickle.dumps(entry))
    env_in.close()
    env_out.close()
    print(f"已生成新的 LMDB 文件：{output_lmdb}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_lmdb", type=str, required=True, help="原始LMDB文件路径")
    parser.add_argument("--output_lmdb", type=str, required=True, help="输出LMDB文件路径")
    parser.add_argument("--target_name", type=str, required=True, help="指定pocket1 name")
    args = parser.parse_args()
    overwrite_pocket1_fields.target_name = args.target_name
    overwrite_pocket1_fields(args.input_lmdb, args.output_lmdb)