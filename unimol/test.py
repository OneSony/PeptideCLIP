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
    if args.test_task=="BCMA":
        task.test_outer("/data/private/ly/CLIP_test/bcma/bcma.lmdb", model)
    elif args.test_task=="HER2":
        task.test_outer("/data/private/ly/CLIP_test/her2/her2.lmdb", model)
    elif args.test_task=="CD38":
        task.test_outer("/data/private/ly/CLIP_test/cd38/cd38.lmdb", model)
    elif args.test_task=="BCMA_same_pocket1":
        task.test_outer("/data/private/ly/CLIP_test/bcma/test/same_pocket1", model)
    elif args.test_task=="BCMA_same_pocket1_021":
        task.test_outer("/data/private/ly/CLIP_test/bcma/test/same_pocket_021_D_B", model)
    elif args.test_task=="BCMA_same_pocket1_050":
        task.test_outer("/data/private/ly/CLIP_test/bcma/test/same_pocket_050_D_B", model)
    elif args.test_task=="BCMA_same_pocket1_078":
        task.test_outer("/data/private/ly/CLIP_test/bcma/test/same_pocket_078_F_L", model)


def cli_main():
    # add args
    

    parser = options.get_validation_parser()
    parser.add_argument("--test-task", type=str, default="BCMA", help="test task")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
