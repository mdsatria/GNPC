import string

from gnpc.default_modules import *


def select_optimiser(model, optim_name: str, lr: float, weight_decay: float):
    if optim_name == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optim_name == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    return optimiser


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, eps=1e-4, method="min") -> None:
        """
        method :
            min = looping stop when there is metric not decreased after num patience
            min = looping stop when there is metric not increased after num patience
        """
        self.patience = patience
        self.verbose = verbose
        self.eps = eps
        self.method = method
        self.counter = 0  # Counter to keep track of epochs without improvement
        self.best_metric = None  # Best validation metric value observed so far
        self.early_stop = False

    def check(self, current_metric):
        triggered = False
        if self.method == "min":
            if current_metric >= self.best_metric + self.eps:
                triggered = True

        if self.method == "max":
            if current_metric <= self.best_metric + self.eps:
                triggered = True

        return triggered

    def __call__(self, current_metric, logger=None):
        is_better = False

        if self.best_metric is None:
            self.best_metric = current_metric
            is_better = True

        else:
            if self.check(current_metric):
                self.counter += 1
                if self.verbose:
                    if logger is None:
                        print(f"    early stop triggerred: {self.counter}")
                    else:
                        logger.info(f"  early stop triggerred: {self.counter}")
            else:
                self.counter = 0
                self.best_metric = current_metric
                is_better = True

        if self.counter >= self.patience:
            self.early_stop = True

        return is_better
