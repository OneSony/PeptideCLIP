#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks
import numpy as np
from tqdm import tqdm
import unicore

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")


#from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve



def main(args):

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)


    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    # Print args
    logger.info(args)


    model.eval()
    
    #names, scores = task.retrieve_mols(model, args.mol_path, args.pocket_path, args.emb_dir, 10000)
    all_results = task.retrieve_pockets(model, args.query_pocket_path, args.target_pocket_path, args.emb_dir, 100, "21")
    #使用一个pocket2(receptor)去检索另一个pocket1(ligand)库
    with open(args.results_path, "w") as f:
        for result in all_results:
            query_name = result['query_name']
            target_pocket_names = result['top_k_names']
            target_pocket_scores = result['top_k_scores']
            
            line = query_name
            for name, score in zip(target_pocket_names, target_pocket_scores):
                line += f"\t{name}:{score:.6f}"
            f.write(line + "\n")
            


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--query-pocket-path", type=str, default="", help="path for pocket1 data")
    parser.add_argument("--target-pocket-path", type=str, default="", help="path for pocket2 data")
    parser.add_argument("--emb-dir", type=str, default="", help="path for saved embedding data")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
