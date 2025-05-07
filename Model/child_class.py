import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
from modeling_PHLF import TransTabLinearClassifier, TransTabLinearClassifier_gated
import numpy as np


class RegularLoss(nn.Module):
    def __init__(self, lambda_reg=0.1):
        super(RegularLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, logits_child, y_child, probs_parent, threshold):
        base_loss = F.cross_entropy(logits_child, y_child, reduction='none')

        child_probs = F.softmax(logits_child, dim=-1)

        conflict1 = (probs_parent.squeeze()<threshold) & (child_probs.argmax(1) != 0)
        conflict2 = (probs_parent.squeeze()>=threshold) & (child_probs.argmax(1) == 0)

        reg_loss = (conflict1.float().sum() + conflict2.float().sum()) /child_probs.size(0)

        total_loss = base_loss + self.lambda_reg * reg_loss
        return total_loss


class GateChildHead(nn.Module):
    def __init__(self,
                 parent_classes,
                 child_classes,
                 hidden_dim=128,
                 lambda_reg=0.5,
                 weights= [1.,5.,15.],
                 device=torch.device('cuda:0')):
        super(GateChildHead, self).__init__()
        self.lambda_reg = lambda_reg
        self.device = device
        self.weights = weights
        self.clf = TransTabLinearClassifier_gated(num_class=child_classes,
                                                  parent_dim = parent_classes,
                                                  hidden_dim=hidden_dim).to(self.device)
        # self.clf = TransTabLinearClassifier(num_class=child_classes,
        #                                           hidden_dim=hidden_dim).to(self.device)

    def RegularLoss(self, logits_child, y_child, probs_parent, threshold):
        weights = torch.tensor(self.weights).to(self.device)
        logits_child = logits_child.float()  # Ensure logits_child is float32
        y_child = y_child.long()  # Ensure y_child is long (integer) type as required by cross_entropy
        base_loss = F.cross_entropy(logits_child, y_child, reduction='none',weight=weights)
        base_loss = base_loss.mean()

        child_probs = F.softmax(logits_child, dim=-1)

        # conflict1 = (probs_parent.squeeze() < threshold) & (child_probs.argmax(1) != 0)
        conflict2 = (probs_parent.squeeze() >= threshold) & (child_probs.argmax(1) == 0)

        reg_loss = conflict2.float().mean()

        total_loss = base_loss + self.lambda_reg * reg_loss
        return total_loss

    def forward(self, x, y, parent_prob, threshold):
        y = y.astype(float)
        y = torch.tensor(np.array(y)).to(self.device)
        parent_prob = torch.tensor(parent_prob).to(self.device)
        x = torch.tensor(x).to(self.device)
        logits = self.clf(x, parent_prob)
        # logits = self.clf(x)
        logits = torch.softmax(logits, -1)
        loss = self.RegularLoss(logits, y, parent_prob, threshold)
        loss = loss.mean()

        return logits, loss

class GateChildHead_eval(nn.Module):
    def __init__(self,
                 parent_classes,
                 child_classes,
                 hidden_dim=128,
                 lambda_reg=0.5,
                 weights= [1.,5.,15.],
                 device=torch.device('cuda:0')):
        super(GateChildHead_eval, self).__init__()
        self.lambda_reg = lambda_reg
        self.device = device
        self.weights = weights
        self.clf = TransTabLinearClassifier_gated(num_class=child_classes,
                                                  parent_dim = parent_classes,
                                                  hidden_dim=hidden_dim).to(self.device)
        # self.clf = TransTabLinearClassifier(num_class=child_classes,
        #                                           hidden_dim=hidden_dim).to(self.device)

    def RegularLoss(self, logits_child, y_child, probs_parent, threshold):
        weights = torch.tensor(self.weights).to(self.device)
        logits_child = logits_child.float()  # Ensure logits_child is float32
        y_child = y_child.long()  # Ensure y_child is long (integer) type as required by cross_entropy
        base_loss = F.cross_entropy(logits_child, y_child, reduction='none',weight=weights)
        base_loss = base_loss.mean()

        child_probs = F.softmax(logits_child, dim=-1)

        # conflict1 = (probs_parent.squeeze() < threshold) & (child_probs.argmax(1) != 0)
        conflict2 = (probs_parent.squeeze() >= threshold) & (child_probs.argmax(1) == 0)

        reg_loss = conflict2.float().mean()

        total_loss = base_loss + self.lambda_reg * reg_loss
        return total_loss

    def forward(self, x, parent_prob, threshold):

        parent_prob = torch.tensor(parent_prob).to(self.device)
        x = torch.tensor(x).to(self.device)
        logits = self.clf(x, parent_prob)
        # logits = self.clf(x)
        logits = torch.softmax(logits, -1)


        return logits







