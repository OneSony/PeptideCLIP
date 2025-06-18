# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import scipy.stats as stats


def calculate_bedroc(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    #print(scores.shape, y_true.shape)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc

@register_loss("peptideclip")
class PocketContrastiveLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            pocket1_list=sample.get("pocket1_name", None),
            pocket2_list=sample.get("pocket2_name", None),
            features_only=True,
            fix_encoder=fix_encoder,
            is_train=self.training
        )
        
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = logit_output.size(0)
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        affinities = sample["target"]["finetune_target"].view(-1)
        
        if not self.training:
            logit_output = logit_output[:,:sample_size]
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": targets,
                "pocket1_name": sample.get("pocket1_name", None),
                "pocket2_name": sample.get("pocket2_name", None),
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "scale": net_output[1].data,
                "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "scale": net_output[1].data
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_pocket1 = F.log_softmax(net_output.float(), dim=-1)
        lprobs_pocket1 = lprobs_pocket1.view(-1, lprobs_pocket1.size(-1))
        sample_size = lprobs_pocket1.size(0)
        targets = torch.arange(sample_size, dtype=torch.long).view(-1).cuda()

        # pocket1 retrieve pocket2 (行方向检索)
        loss_pocket1 = F.nll_loss(
            lprobs_pocket1,
            targets,
            reduction="sum" if reduce else "none",
        )
        
        lprobs_pocket2 = F.log_softmax(torch.transpose(net_output.float(), 0, 1), dim=-1)
        lprobs_pocket2 = lprobs_pocket2.view(-1, lprobs_pocket2.size(-1))
        lprobs_pocket2 = lprobs_pocket2[:sample_size]

        # pocket2 retrieve pocket1 (列方向检索)
        loss_pocket2 = F.nll_loss(
            lprobs_pocket2,
            targets,
            reduction="sum" if reduce else "none",
        )
        
        # 双向对比学习损失
        loss = 0.5 * loss_pocket1 + 0.5 * loss_pocket2
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            
            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs)-1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)
            
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )

            metrics.log_scalar(
                "valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            
            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )
            print(f"Pocket similarity - targets.shape: {targets.shape}, probs.shape: {probs.shape}")

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target] = 1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)
            
            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_pocket_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_pocket_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_pocket_top3_acc", top_k_acc, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

