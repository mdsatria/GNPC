from copy import deepcopy

from gnpc.default_modules import *
from gnpc.survival.analysis.evaluation import PatientStratification


class TrainGNNSurvival:
    def __init__(
        self,
        model,
        optimiser,
        criterion,
        device,
        best_model=None,
        scheduler=None,
        clip_gradient_thresh=1,
        is_discreet=False,
    ):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.clip_gradient_thresh = clip_gradient_thresh
        self.best_model = best_model
        self.is_discreet = is_discreet
        self.model.to(self.device)

    def set_best_model(self):
        self.best_model = deepcopy(self.model)

    def train(self, current_epoch, dloader):  # coxph or deepsurv
        losses = []
        y_patient = []
        y_time = np.array([])
        y_event = np.array([])
        score = np.array([])

        pbar = tqdm(dloader, bar_format="{l_bar}{bar:40}")

        self.model.train()
        for ix_bar, data in enumerate(pbar):

            if len(data["time"]) > 1:  # skip training if only 1 data

                self.optimiser.zero_grad()

                graph = data["graph"].to(self.device)
                batch_event = data["event"].to(self.device)
                batch_time = data["time"].to(self.device)
                # batch_label = data["time_label"].to(self.device)
                batch_clinical = data["clinical_data"].to(self.device)
                batch_patient = data["patient"]

                if self.is_discreet:
                    pass
                    hazards_prob, survival_prob, Y_hat, _ = self.model(
                        graph, batch_clinical
                    )
                    loss = self.criterion(
                        hazards=hazards_prob,
                        S=survival_prob,
                        # Y=torch.unsqueeze(batch_label, dim=1),
                        c=torch.unsqueeze(batch_event, dim=1),
                    )
                    logits = -torch.sum(survival_prob, dim=1)  # .detach().cpu().numpy()
                else:
                    logits, _ = self.model(graph, batch_clinical)
                    loss = self.criterion(logits, batch_time, batch_event, self.model)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_gradient_thresh
                )
                self.optimiser.step()

                losses.append(loss.item())
                y_patient.extend(batch_patient)
                y_time = np.concatenate(
                    (y_time, batch_time.ravel().detach().cpu().numpy())
                )
                y_event = np.concatenate(
                    (y_event, batch_event.ravel().detach().cpu().numpy())
                )
                score = np.concatenate((score, logits.ravel().detach().cpu().numpy()))
                current_lr = self.optimiser.param_groups[0]["lr"]
                pbar.set_description(
                    f"train loss {ix_bar+1}/{len(pbar)}: {loss.item():.3f} lr:{current_lr:.2e}"
                )
        if (self.scheduler is not None) and (current_epoch > 10):
            self.scheduler.step()

        losses = sum(losses) / len(losses)
        cut_off, ps = self._scoring(score, y_patient, y_time, y_event)

        # return losses, ps.c_index, ps.p_value, ps, cut_off

        return losses, ps, cut_off

    def infer(self, dloader, cut_off_in=np.inf, is_train=False, use_best_model=False):

        if use_best_model:
            self.best_model.eval()
        else:
            self.model.eval()

        losses = []
        y_patient = []
        y_time = np.array([])
        y_event = np.array([])
        score = np.array([])

        pbar = tqdm(dloader, bar_format="{l_bar}{bar:40}")
        with torch.inference_mode():
            for ix_bar, data in enumerate(pbar):

                graph = data["graph"].to(self.device)
                batch_event = data["event"].to(self.device)
                batch_time = data["time"].to(self.device)
                batch_clinical = data["clinical_data"].to(self.device)
                batch_patient = data["patient"]

                if use_best_model:
                    self.best_model.to(self.device)
                    logits, _ = self.best_model(graph, batch_clinical)
                    loss = self.criterion(
                        logits, batch_time, batch_event, self.best_model
                    )
                else:
                    logits, _ = self.model(graph, batch_clinical)
                    loss = self.criterion(logits, batch_time, batch_event, self.model)
                losses.append(loss.item())
                y_patient.extend(batch_patient)
                y_time = np.concatenate(
                    (y_time, batch_time.ravel().detach().cpu().numpy())
                )
                y_event = np.concatenate(
                    (y_event, batch_event.ravel().detach().cpu().numpy())
                )
                score = np.concatenate((score, logits.ravel().detach().cpu().numpy()))
                pbar.set_description(
                    f"test loss {ix_bar+1}/{len(pbar)}: {loss.item():.3f}"
                )

        losses = sum(losses) / len(losses)

        if is_train:
            cut_off_in = None

        cut_off, ps = self._scoring(score, y_patient, y_time, y_event, cut_off_in)

        # return losses, ps.c_index, ps.p_value, ps, cut_off

        return losses, ps, cut_off

    def _scoring(self, score, y_patient, y_time, y_event, cut_off_in=None):
        # score = -score
        df_prediction = pd.DataFrame(
            {"patient": y_patient, "time": y_time, "event": y_event, "score": score}
        )
        score_per_patient = np.array(
            df_prediction.groupby("patient")["score"].mean().values.tolist()
        )
        if cut_off_in is None:
            cut_off = np.median(score_per_patient)
        else:
            cut_off = cut_off_in
        ps = PatientStratification(
            df=df_prediction,
            cut_off=cut_off,
        )

        return cut_off, ps

    def update_model(self, losses, loss):
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_gradient_thresh
        )
        self.optimiser.step()
