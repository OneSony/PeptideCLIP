import lmdb
import pickle
import sys

def merge_lmdb(lmdb_paths, output_path):
    env_out = lmdb.open(output_path, map_size=int(1e12), subdir=False, readonly=False)
    txn_out = env_out.begin(write=True)
    idx = 0
    for lmdb_path in lmdb_paths:
        env_in = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with env_in.begin() as txn_in:
            cursor = txn_in.cursor()
            for key, value in cursor:
                txn_out.put(str(idx).encode('ascii'), value)
                idx += 1
        env_in.close()
    txn_out.commit()
    env_out.close()
    print(f"Merged {len(lmdb_paths)} LMDBs into {output_path}, total {idx} entries.")

if __name__ == "__main__":
    # 用法示例：python merge_lmdb.py train1.lmdb train2.lmdb merged_train.lmdb
    if len(sys.argv) < 4:
        print("Usage: python merge_lmdb.py lmdb1 lmdb2 ... output_lmdb")
        sys.exit(1)
    lmdb_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    merge_lmdb(lmdb_paths, output_path)