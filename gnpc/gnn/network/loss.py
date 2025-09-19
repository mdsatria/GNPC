import numpy as np
import torch
import torch.nn as nn


class RankMSEBCE(nn.Module):
    def __init__(
        self,
        device="cuda",
        weight=None,
    ):
        super(RankMSEBCE, self).__init__()
        self.device = device
        if weight is None:
            self.weight = {"rank": 0.4, "bce": 0.3, "mse": 0.3}
        else:
            self.weight = weight

    def forward(
        self,
        output1,
        output2,
        target_rank,
        pred_event,
        gt_event,
        pred_time,
        gt_time,
    ):
        bce_mean = torch.nn.functional.binary_cross_entropy(pred_event, gt_event)
        mse_mean = torch.nn.functional.mse_loss(pred_time, gt_time)

        margin_loss = torch.nn.functional.margin_ranking_loss(
            input1=output1, input2=output2, target=target_rank, margin=0
        )

        return (
            (self.weight["rank"] * margin_loss)
            + (self.weight["bce"] * bce_mean)
            + (self.weight["mse"] * mse_mean)
        )


"""
DeepSurv Loss
https://github.com/czifan/DeepSurv.pytorch
"""


class Regularization(object):
    def __init__(self, order, weight_decay):
        """The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        """
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        """Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        """
        reg_loss = 0
        for name, w in model.named_parameters():
            if "weight" in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


def PartialLogLikelihood(logits, fail_indicator, ties):
    """
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model
    ties: 'noties' or 'efron' or 'breslow'
    """
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == "noties":
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0 * logL / observations


class NLLDeepSurv(nn.Module):  # NegativeLogLikelihood
    def __init__(self, l2=1e-5, device="cuda"):
        super(NLLDeepSurv, self).__init__()
        self.l2 = l2
        self.device = device
        self.reg = Regularization(order=2, weight_decay=self.l2)

    def forward(self, risk_pred, T, E, model):
        # index_sort = torch.argsort(T)
        # T = T[index_sort]
        # E = E[index_sort]
        # risk_pred = risk_pred[index_sort]

        current_batch_len = len(risk_pred)
        R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = torch.tensor(R_matrix_train, dtype=torch.float32)
        train_R = train_R.to(self.device)
        # train_ystatus = torch.FloatTensor(E).cuda()

        theta = risk_pred.reshape(-1)
        # print(theta)
        exp_theta = torch.exp(theta)
        # print(exp_theta)

        loss_nn = -torch.mean(
            (theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * E
        )
        # print(loss_nn)
        l2_loss = self.reg(model)
        return loss_nn + l2_loss


def nll_loss_cat(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(
            1 - hazards, dim=1
        )  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat(
        [torch.ones_like(c), S], 1
    )  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLossCat(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss_cat(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss_cat(hazards, S, Y, c, alpha=alpha)
