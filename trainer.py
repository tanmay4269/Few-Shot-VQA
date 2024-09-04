import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import *
from models.models import *
from Trainer import *
from utils import *
import config

import optuna

# Domain Adaptation Trainer
class DATrainer(Trainer):
    def __init__(self, cfg, vqa_v2, vqa_abs):
        cfg["embedder_lr"] = cfg["base_lr"]
        cfg["classifier_lr"] = cfg["base_lr"]

        super().__init__(cfg, vqa_v2, vqa_abs)

        self.criterion_label = nn.CrossEntropyLoss(reduction="none")
        self.criterion_label_type = nn.CrossEntropyLoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

    def init_model(self):
        return DANN_VLModel(self.cfg).cuda()

    def init_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.cfg['base_lr'], 
            weight_decay=self.cfg['weight_decay'],
        )

        return optimizer

    def init_dataloader(self):
        # Init Data
        (v2_train_data, v2_val_data), (abs_train_data, abs_val_data) = (
            data_processing_v2(self.cfg, vqa_v2, vqa_abs)
        )

        n_v2_t, n_abs_t = len(v2_train_data), len(abs_train_data)
        n_v2_v, n_abs_v = len(v2_val_data), len(abs_val_data)

        n_ratio_t = n_v2_t / n_abs_t
        n_ratio_v = n_v2_v / n_abs_v

        if n_ratio_t > 1.0:
            abs_train_data = abs_train_data * ceil(n_ratio_t)
            abs_train_data = abs_train_data[:n_v2_t]

            abs_val_data = abs_val_data * ceil(n_ratio_v)
            abs_val_data = abs_val_data[:n_v2_v]
        else:
            v2_train_data = v2_train_data * ceil(n_ratio_t)
            v2_train_data = v2_train_data[:n_abs_t]

            v2_val_data = v2_val_data * ceil(n_ratio_v)
            v2_val_data = v2_val_data[:n_abs_v]

        # Init DataLoader
        v2_train_dataset = VQADataset(self.cfg, v2_train_data)
        v2_val_dataset = VQADataset(self.cfg, v2_val_data)

        abs_train_dataset = VQADataset(self.cfg, abs_train_data)
        abs_val_dataset = VQADataset(self.cfg, abs_val_data)

        # debug
        # shuffle = False
        shuffle = True

        self.v2_train_dataloader = DataLoader(
            v2_train_dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle
        )

        self.v2_val_dataloader = DataLoader(
            v2_val_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

        self.abs_train_dataloader = DataLoader(
            abs_train_dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle
        )

        self.abs_val_dataloader = DataLoader(
            abs_val_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

        self.num_train_batches = len(self.v2_train_dataloader)
        self.num_val_batches = len(self.v2_val_dataloader)

        self.v2_domain_label = torch.tensor(0).float()
        self.abs_domain_label = torch.tensor(1).float()

    def plot(self, epoch):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        train_losses = np.array(self.train_losses)

        ax[0].plot(np.arange(epoch), train_losses[:, 0], label="Train Label Loss")
        ax[0].plot(np.arange(epoch), train_losses[:, 1], label="Train Domain Loss")
        ax[0].plot(np.arange(epoch), train_losses[:, 2], label="Train Label Type Loss")
        ax[0].plot(np.arange(epoch), train_losses[:, 3], label="Train Loss")
        ax[0].plot(np.arange(epoch), self.eval_losses, label="Eval Loss")
        ax[0].legend()
        ax[0].set_title("Losses")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        accuracies = np.array(self.accuracies)
        # ax[1].plot(np.arange(epoch), accuracies[:, 0], label="V2 Accuracy")
        # ax[1].plot(np.arange(epoch), accuracies[:, 1], label="Abs Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 2], label="Label Type Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 3], label="Label Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 4], label="Domain Accuracy")
        ax[1].legend()
        ax[1].set_title("Accuracies")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")

        fig.suptitle(self.cfg["title"])
        plt.show()

    def get_alpha(self, i_dataloader, len_dataloader):
        p = (
            float(i_dataloader + self.epoch * len_dataloader)
            / self.num_epochs
            / len_dataloader
        )
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1

    def get_loss(
        self,
        v2_label_logits,
        v2_domain_logits,
        v2_label_type_logits,
        abs_label_logits,
        abs_domain_logits,
        abs_label_type_logits,
        v2_label,
        abs_label,
        v2_domain_label,
        abs_domain_label,
        v2_label_type,
        abs_label_type,
    ):

        v2_label_loss = self.criterion_label(v2_label_logits, v2_label)
        abs_label_loss = self.criterion_label(abs_label_logits, abs_label)

        v2_domain_loss = self.criterion_domain(v2_domain_logits, v2_domain_label)
        abs_domain_loss = self.criterion_domain(abs_domain_logits, abs_domain_label)
        domain_loss = 0.5 * v2_domain_loss + 0.5 * abs_domain_loss

        v2_label_type_loss = self.criterion_label_type(v2_label_type_logits, v2_label_type)
        abs_label_type_loss = self.criterion_label_type(
            abs_label_type_logits, abs_label_type
        )
        label_type_loss = 0.5 * v2_label_type_loss + 0.5 * abs_label_type_loss

        v2_domain_prob = F.sigmoid(v2_domain_logits)
        abs_domain_prob = F.sigmoid(abs_domain_logits)

        if self.cfg["domain_adaptation_method"] == "importance_sampling":
            if self.source_domain == "v2":
                v2_weight = 1 - v2_domain_prob
                abs_weight = abs_domain_prob
            elif self.source_domain == "abs":
                v2_weight = v2_domain_prob
                abs_weight = 1 - abs_domain_prob
        else:
            v2_weight = 1.0
            abs_weight = 1.0

        weighted_v2_label_loss = (v2_weight * v2_label_loss).mean()
        weighted_abs_label_loss = (abs_weight * abs_label_loss).mean()

        label_loss = 0.5 * weighted_v2_label_loss + 0.5 * weighted_abs_label_loss

        if self.cfg['use_label_type_classifier']:
            total_loss = (
                0.5 * (0.5 * label_loss + 0.5 * label_type_loss) + 0.5 * domain_loss
            )
        else:
            total_loss = 0.5 * label_loss + 0.5 * domain_loss

        return label_loss, domain_loss, label_type_loss, total_loss

    def get_accuracy(
        self,
        v2_label_logits,
        v2_domain_logits,
        v2_label_type_logits,
        abs_label_logits,
        abs_domain_logits,
        abs_label_type_logits,
        v2_label,
        abs_label,
        v2_domain_label,
        abs_domain_label,
        v2_label_type,
        abs_label_type,
    ):
        # V2 - label
        _, v2_predicted_indices = torch.max(v2_label_logits, dim=1)
        v2_label_indices = torch.argmax(v2_label, dim=1)
        is_correct_v2 = v2_predicted_indices == v2_label_indices

        v2_total = v2_label.shape[0]
        v2_correct = is_correct_v2.sum().item()

        # V2 - label type
        _, v2_predicted_type_indices = torch.max(v2_label_type_logits, dim=1)
        v2_label_type_indices = torch.argmax(v2_label_type, dim=1)
        is_correct_type_v2 = v2_predicted_type_indices == v2_label_type_indices

        v2_correct_type = is_correct_type_v2.sum().item()

        # Abs - label
        _, abs_predicted_indices = torch.max(abs_label_logits, dim=1)
        abs_label_indices = torch.argmax(abs_label_type, dim=1)
        is_correct_abs = abs_predicted_indices == abs_label_indices

        abs_total = abs_label.shape[0]
        abs_correct = is_correct_abs.sum().item()

        # Abs - label type
        _, abs_predicted_type_indices = torch.max(abs_label_type_logits, dim=1)
        abs_label_type_indices = torch.argmax(abs_label, dim=1)
        is_correct_type_abs = abs_predicted_type_indices == abs_label_type_indices

        abs_correct_type = is_correct_type_abs.sum().item()

        # Total - label
        total = v2_label.shape[0] + abs_label.shape[0]
        correct = is_correct_v2.sum().item() + is_correct_abs.sum().item()

        correct_type = (
            is_correct_type_v2.sum().item() + is_correct_type_abs.sum().item()
        )

        # domain
        v2_domain_pred = F.sigmoid(v2_domain_logits) > 0.5
        abs_domain_pred = F.sigmoid(abs_domain_logits) > 0.5
        is_correct_v2_d = v2_domain_pred == v2_domain_label
        is_correct_abs_d = abs_domain_pred == abs_domain_label

        domain_total = v2_domain_label.shape[0] + abs_domain_label.shape[0]
        domain_correct = is_correct_v2_d.sum().item() + is_correct_abs_d.sum().item()

        accuracies = [
            v2_total,
            v2_correct,
            abs_total,
            abs_correct,
            total,
            correct,
            domain_total,
            domain_correct,
            v2_correct_type,
            abs_correct_type,
            correct_type,
        ]

        return accuracies

    def process_input(
        self,
        v2_i_tokens,
        v2_q_tokens,
        v2_label,
        v2_label_type,
        abs_i_tokens,
        abs_q_tokens,
        abs_label,
        abs_label_type,
        alpha,
    ):
        v2_i_tokens = {key: value.cuda() for key, value in v2_i_tokens.items()}
        v2_q_tokens = {key: value.cuda() for key, value in v2_q_tokens.items()}

        abs_i_tokens = {key: value.cuda() for key, value in abs_i_tokens.items()}
        abs_q_tokens = {key: value.cuda() for key, value in abs_q_tokens.items()}

        v2_logits = self.model(v2_i_tokens, v2_q_tokens, alpha)
        abs_logits = self.model(abs_i_tokens, abs_q_tokens, alpha)

        v2_label, abs_label = v2_label.cuda(), abs_label.cuda()
        v2_label_type, abs_label_type = v2_label_type.cuda(), abs_label_type.cuda()

        v2_domain_label = self.v2_domain_label.repeat(v2_label.shape[0], 1).cuda()
        abs_domain_label = self.abs_domain_label.repeat(abs_label.shape[0], 1).cuda()

        v2_label_logits, v2_domain_logits, v2_label_type_logits = v2_logits
        abs_label_logits, abs_domain_logits, abs_label_type_logits = abs_logits

        args = [
            v2_label_logits,
            v2_domain_logits,
            v2_label_type_logits,
            abs_label_logits,
            abs_domain_logits,
            abs_label_type_logits,
            v2_label,
            abs_label,
            v2_domain_label,
            abs_domain_label,
            v2_label_type,
            abs_label_type,
        ]

        losses = self.get_loss(*args)

        accuracies = self.get_accuracy(*args)

        return (losses, accuracies)

    def train_epoch(self):
        self.model.train()
        label_loss_meter = AverageMeter()
        domain_loss_meter = AverageMeter()
        label_type_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()

        for i, (
            (v2_i_tokens, v2_q_tokens, v2_label, v2_label_type),
            (abs_i_tokens, abs_q_tokens, abs_label, abs_label_type),
        ) in enumerate(self.train_dataloader):

            self.alpha = self.get_alpha(i, self.num_train_batches)

            (label_loss, domain_loss, label_type_loss, total_loss), _ = (
                self.process_input(
                    v2_i_tokens,
                    v2_q_tokens,
                    v2_label,
                    v2_label_type,
                    abs_i_tokens,
                    abs_q_tokens,
                    abs_label,
                    abs_label_type,
                    self.alpha,
                )
            )

            label_loss_meter.update(label_loss.item())
            domain_loss_meter.update(domain_loss.item())
            label_type_loss_meter.update(label_type_loss.item())
            total_loss_meter.update(total_loss.item())

            # if (
            #     self.cfg["print_logs"]
            #     and self.num_train_batches > 4
            #     and i % (self.num_train_batches // 4) == 0
            # ):
            #     print(
            #         f"\t Iter [{i}/{self.num_train_batches}]\t Loss: {total_loss.item():.6f}"
            #     )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # debug
            # break

        return (
            label_loss_meter.avg,
            domain_loss_meter.avg,
            label_type_loss_meter.avg,
            total_loss_meter.avg,
        )

    def eval_epoch(self):
        v2_correct, v2_total = 0, 0
        abs_correct, abs_total = 0, 0
        label_type_correct = 0
        correct, total = 0, 0
        domain_correct, domain_total = 0, 0

        running_loss = 0.0

        for i, (
            (v2_i_tokens, v2_q_tokens, v2_label, v2_label_type),
            (abs_i_tokens, abs_q_tokens, abs_label, abs_label_type),
        ) in enumerate(self.val_dataloader):
            (
                (label_loss, domain_loss, label_type_loss, total_loss),
                (
                    _v2_total,
                    _v2_correct,
                    _abs_total,
                    _abs_correct,
                    _total,
                    _correct,
                    _domain_total,
                    _domain_correct,
                    _v2_correct_type,
                    _abs_correct_type,
                    _correct_type,
                ),
            ) = self.process_input(
                v2_i_tokens,
                v2_q_tokens,
                v2_label,
                v2_label_type,
                abs_i_tokens,
                abs_q_tokens,
                abs_label,
                abs_label_type,
                self.alpha,
            )

            v2_total += _v2_total
            v2_correct += _v2_correct

            abs_total += _abs_total
            abs_correct += _abs_correct

            label_type_correct += _correct_type

            total += _total
            correct += _correct

            domain_total += _domain_total
            domain_correct += _domain_correct

            running_loss += total_loss.item()

            # debug
            # break


        eval_loss = running_loss / self.num_val_batches
        v2_accuracy = v2_correct / v2_total
        abs_accuracy = abs_correct / abs_total
        label_type_accuracy = label_type_correct / total

        total_accuracy = correct / total

        domain_accuracy = domain_correct / domain_total

        return (
            eval_loss, 
            v2_accuracy, 
            abs_accuracy, 
            label_type_accuracy, 
            total_accuracy, 
            domain_accuracy
        )

    def train(self, show_plot, optuna_trial=None, comet_expt=None):
        min_eval_loss = float("inf")
        high_eval_loss_count = 0

        for self.epoch in range(self.num_epochs):
            self.train_dataloader = zip(
                self.v2_train_dataloader, self.abs_train_dataloader
            )
            
            # debug
            # self.val_dataloader = zip(self.v2_train_dataloader, self.abs_train_dataloader)
            self.val_dataloader = zip(self.v2_val_dataloader, self.abs_val_dataloader)

            label_loss, domain_loss, label_type_loss, total_loss = self.train_epoch()
            with torch.no_grad():
                (
                    eval_loss,
                    v2_accuracy,
                    abs_accuracy,
                    label_type_accuracy,
                    total_accuracy,
                    domain_accuracy,
                ) = self.eval_epoch()

            self.scheduler.step()

            self.train_losses.append((label_loss, domain_loss, label_type_loss, total_loss))
            self.eval_losses.append(eval_loss)
            self.accuracies.append(
                (v2_accuracy, abs_accuracy, label_type_accuracy, total_accuracy, domain_accuracy)
            )
            
            if not self.cfg['use_label_type_classifier']:
                label_type_loss = -1
                label_type_accuracy = -1

            # Plotting
            if show_plot and self.epoch > 0 and self.epoch % 10 == 0:
                self.plot(self.epoch + 1)

            # Logging
            if self.cfg["print_logs"]:
                print(
                    f"Epoch [{self.epoch + 1}/{self.num_epochs}]\t \
                        Avg Train Loss: {total_loss:.6f}\t \
                        Avg Eval Loss: {eval_loss:.6f}\t \
                        Avg Domain Accuracy: {domain_accuracy:.2f}\t \
                        Avg Label Type Accuracy: {label_type_accuracy:.2f}\t \
                        Avg Total Eval Accuracy: {total_accuracy:.2f}"
                )

            # Comet Logging
            if comet_expt:
                comet_expt.log_metrics(
                    {
                        "Loss/train_label": label_loss,
                        "Loss/train_domain": domain_loss,
                        "Loss/train_label_type": label_type_loss,
                        "Loss/train_total": total_loss,
                        "Loss/Eval": eval_loss,
                        "EvalAccuracy/domain": domain_accuracy,
                        "EvalAccuracy/v2_label": v2_accuracy,
                        "EvalAccuracy/abs_label": abs_accuracy,
                        "EvalAccuracy/label_type": label_type_accuracy,
                        "EvalAccuracy/avg_label": total_accuracy,
                    },
                    step=self.epoch,
                )

            # Optuna Logging
            if optuna_trial:
                optuna_trial.report(eval_loss, self.epoch)

                if optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Saving model
            if self.epoch > 0 and self.eval_losses[-1] < min_eval_loss:
                min_eval_loss = self.eval_losses[-1]
                torch.save(self.model.state_dict(), self.cfg["weights_save_path"])

            # Early Stopping
            if optuna_trial is None or self.cfg["relaxation_period"] < 0:
                continue

            if self.eval_losses[-1] > self.eval_losses[-2]:
                high_eval_loss_count += 1
                if high_eval_loss_count >= self.cfg["relaxation_period"]:
                    if show_plot:
                        self.plot(self.epoch + 1)
                    break
            else:
                high_eval_loss_count = 0

        # Post Training Plot and Model Upload
        if show_plot:
            self.plot(self.epoch + 1)

        if comet_expt:
            # log_model(comet_expt, self.model, 'da-vqa')
            pass

        return eval_loss


# Only for debugging
if __name__ == "__main__":
    cfg = config.cfg

    trainer = DATrainer(cfg, vqa_v2, vqa_abs)
    trainer.train(show_plot=True)