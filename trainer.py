import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import *
from models.models import *

import optuna
from comet_ml.integration.pytorch import log_model


vqa_v2 = {
    "type": "v2",
    "image_root": "data/vqa-v2/val2014/val2014/COCO_val2014_000000",
    "questions_path": "data/vqa-v2/v2_OpenEnded_mscoco_val2014_questions.json",
    "annotations_path": "data/vqa-v2/v2_mscoco_val2014_annotations.json",
}

vqa_abs = {
    "type": "abs",
    "image_root": "data/vqa-abstract/img_train/abstract_v002_train2015_0000000",
    "questions_path": "data/vqa-abstract/questions_train/OpenEnded_abstract_v002_train2015_questions.json",
    "annotations_path": "data/vqa-abstract/annotations_train/abstract_v002_train2015_annotations.json",
}


class Trainer:
    def __init__(self, cfg, vqa_v2, vqa_abs):
        self.update_cfg(cfg)

        self.num_epochs = cfg["epochs"]
        self.vqa_v2, self.vqa_abs = vqa_v2, vqa_abs
        self.source_domain = cfg["source_domain"]

        self.init_dataloader()
        self.model = self.init_model()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.init_optimizer()
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=self.num_epochs, power=0.9
        )

        # Logging
        self.train_losses, self.eval_losses, self.accuracies = [], [], []

    def init_model(self):
        return VLModel(self.cfg).cuda()

    def init_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg["base_lr"],
            weight_decay=self.cfg["weight_decay"],
        )

    def update_cfg(self, cfg):
        self.cfg = cfg

        title = ""
        for k in [
            "name",
            "n_classes",
            "v2_samples_per_answer",
            "abs_samples_per_answer",
            "source_domain",
            "base_lr",
            # "domain_adaptation_method",
        ]:
            v = self.cfg[k]
            if k != "base_lr": 
                title += f"{k}={v}__"
            else:
                title += f"{k}={v:.2e}__"


        def random_string(n):
            chars = np.array(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
            return ''.join(np.random.choice(chars, n))
        
        self.cfg["title"] = title.replace(" ", "_") + random_string(8)

        self.cfg["weights_save_path"] = (
                self.cfg["weights_save_root"] + "/" + self.cfg["title"] + ".pth"
        )

        if cfg['print_logs']:
            print("weights_save_path:", self.cfg["weights_save_path"])

    def init_dataloader(self):
        (v2_train_data, v2_val_data), (abs_train_data, abs_val_data), labels = (
            data_processing_v2(self.cfg, vqa_v2, vqa_abs)
        )

        if self.source_domain == "v2":
            self.cfg = self.update_cfg(vqa_v2, self.cfg)
            train_data, val_data = v2_train_data, v2_val_data
        elif self.source_domain == "abs":
            self.cfg = self.update_cfg(vqa_abs, self.cfg)
            train_data, val_data = abs_train_data, abs_val_data

        train_dataset = VQADataset(self.cfg, train_data)
        val_dataset = VQADataset(self.cfg, val_data)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.cfg["batch_size"], shuffle=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

    def plot(self, epoch):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(np.arange(epoch), self.train_losses, label="Train Loss")
        ax[0].plot(np.arange(epoch), self.eval_losses, label="Eval Loss")
        ax[0].legend()
        ax[0].set_title("Losses")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].plot(np.arange(epoch), self.accuracies, label="Accuracy")
        ax[1].legend()
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")

        fig.suptitle(self.cfg["title"])
        plt.show()

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        num_batches = len(self.train_dataloader)

        for i, (i_tokens, q_tokens, label) in enumerate(self.train_dataloader):
            i_tokens = {key: value.cuda() for key, value in i_tokens.items()}
            q_tokens = {key: value.cuda() for key, value in q_tokens.items()}
            label = label.cuda()

            logits = self.model(i_tokens, q_tokens)

            loss = self.criterion(logits, label)
            running_loss += loss.item()

            if self.cfg['print_logs'] and num_batches > 16 and i % (num_batches // 4) == 0:
                print(f"\t Iter [{i}/{num_batches}]\t Loss: {loss.item():.6f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = running_loss / num_batches

        return avg_loss

    def eval_epoch(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for i_tokens, q_tokens, label in self.val_dataloader:
                i_tokens = {key: value.cuda() for key, value in i_tokens.items()}
                q_tokens = {key: value.cuda() for key, value in q_tokens.items()}
                label = label.cuda()

                logits = self.model(i_tokens, q_tokens)

                loss = self.criterion(logits, label)
                running_loss += loss.item()

                # Compute accuracy
                _, predicted_indices = torch.max(logits, dim=1)
                label_indices = torch.argmax(label, dim=1)
                is_correct = predicted_indices == label_indices

                total += label.shape[0]
                correct += is_correct.sum().item()

        avg_loss = running_loss / len(self.val_dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, show_plot):
        max_accuracy = 0.0
        low_accuracy_count = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            eval_loss, accuracy = self.eval_epoch()
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.eval_losses.append(eval_loss)
            self.accuracies.append(accuracy)

            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}]\t Avg Train Loss: {train_loss:.6f}\t Avg Eval Loss: {eval_loss:.6f}\t Avg Eval Accuracy: {accuracy:.2f}"
            )

            if show_plot and epoch > 0 and epoch % 10 == 0:
                self.plot(epoch + 1)

            if epoch == 0:
                continue

            if self.accuracies[-1] > max_accuracy:
                max_accuracy = self.accuracies[-1]
                torch.save(self.model.state_dict(), self.cfg["weights_save_path"])

            if self.accuracies[-1] < self.accuracies[-2]:
                low_accuracy_count += 1
                if low_accuracy_count >= self.cfg["relaxation_period"]:
                    if show_plot:
                        self.plot(epoch + 1)
                    break
            else:
                low_accuracy_count = 0


# Domain Adaptation Trainer
class DA_Trainer(Trainer):
    def __init__(self, cfg, vqa_v2, vqa_abs):
        cfg["embedder_lr"] = cfg["base_lr"]
        cfg["label_classifier_lr"] = cfg["base_lr"]
        cfg["domain_classifier_lr"] = cfg["base_lr"]

        super().__init__(cfg, vqa_v2, vqa_abs)

        self.criterion_label = nn.CrossEntropyLoss(reduction="none")
        self.criterion_domain = nn.BCEWithLogitsLoss()

    def init_model(self):
        return VLModel_IS(self.cfg).cuda()

    def init_optimizer(self):
        embedder_params = [
            p
            for name, p in self.model.named_parameters()
            if ("label_classifier" not in name) and ("domain_classifier" not in name)
        ]

        param_groups = [
            {
                "params": embedder_params,
                "lr": self.cfg["embedder_lr"],
                "weight_decay": self.cfg["weight_decay"],
            },
            {
                "params": self.model.label_classifier.parameters(),
                "lr": self.cfg["label_classifier_lr"],
                "weight_decay": self.cfg["weight_decay"],
            },
            {
                "params": self.model.domain_classifier.parameters(),
                "lr": self.cfg["domain_classifier_lr"],
                "weight_decay": self.cfg["weight_decay"],
            },
        ]

        optimizer = torch.optim.Adam(param_groups)

        return optimizer

    def init_dataloader(self):
        # Init Data
        (v2_train_data, v2_val_data), (abs_train_data, abs_val_data), labels = (
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

        self.v2_train_dataloader = DataLoader(
            v2_train_dataset, batch_size=self.cfg["batch_size"], shuffle=True
        )

        self.v2_val_dataloader = DataLoader(
            v2_val_dataset, batch_size=self.cfg["batch_size"], shuffle=False
        )

        self.abs_train_dataloader = DataLoader(
            abs_train_dataset, batch_size=self.cfg["batch_size"], shuffle=True
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

        ax[0].plot(np.arange(epoch), train_losses[:, 0], label="Label Loss")
        ax[0].plot(np.arange(epoch), train_losses[:, 1], label="Domain Loss")
        ax[0].plot(np.arange(epoch), train_losses[:, 2], label="Train Loss")
        ax[0].plot(np.arange(epoch), self.eval_losses, label="Eval Loss")
        ax[0].legend()
        ax[0].set_title("Losses")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        accuracies = np.array(self.accuracies)
        ax[1].plot(np.arange(epoch), accuracies[:, 0], label="V2 Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 1], label="Abs Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 2], label="Total Accuracy")
        ax[1].plot(np.arange(epoch), accuracies[:, 3], label="Domain Accuracy")
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
            abs_label_logits,
            abs_domain_logits,
            v2_label,
            abs_label,
            v2_domain_label,
            abs_domain_label,
            
            v2_label_type_logits=None,
            abs_label_type_logits=None,
            v2_label_type=None,
            abs_label_type=None,
    ):
        
        v2_label_loss = self.criterion_label(v2_label_logits, v2_label)
        abs_label_loss = self.criterion_label(abs_label_logits, abs_label)

        v2_domain_loss = self.criterion_domain(v2_domain_logits, v2_domain_label)
        abs_domain_loss = self.criterion_domain(abs_domain_logits, abs_domain_label)
        domain_loss = 0.5 * v2_domain_loss + 0.5 * abs_domain_loss

        if v2_label_type:
            v2_label_type_loss = self.criterion_label(v2_label_type_logits, v2_label_type)
            abs_label_type_loss = self.criterion_label(abs_label_type_logits, abs_label_type)
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

        if v2_label_type:
            total_loss = 0.5 * (0.5 * label_loss + 0.5 * label_type_loss) + 0.5 * domain_loss
        else:
            total_loss = 0.5 * label_loss + 0.5 * domain_loss

        if not self.cfg['use_label_type_classifier']:
            return label_loss, domain_loss, total_loss
        else:
            return label_loss, domain_loss, label_type_loss, total_loss

    def get_accuracy(
            self,
            v2_label_logits,
            v2_domain_logits,
            abs_label_logits,
            abs_domain_logits,
            v2_label,
            abs_label,
            v2_domain_label,
            abs_domain_label,
            
            v2_label_type_logits=None,
            abs_label_type_logits=None,
            v2_label_type=None,
            abs_label_type=None,
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

        correct_type = is_correct_type_v2.sum().item() + is_correct_type_abs.sum().item()

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
        ]

        if self.cfg['use_label_type_classifier']:
            type_accuracies = [
                v2_correct_type,
                abs_correct_type,
                correct_type,
            ]

            accuracies += type_accuracies

        return accuracies

    def process_input(
            self,
            v2_i_tokens,
            v2_q_tokens,
            v2_label,
            abs_i_tokens,
            abs_q_tokens,
            abs_label,
            alpha,

            v2_label_type=None,
            abs_label_type=None,
    ):
        v2_i_tokens = {key: value.cuda() for key, value in v2_i_tokens.items()}
        v2_q_tokens = {key: value.cuda() for key, value in v2_q_tokens.items()}

        abs_i_tokens = {key: value.cuda() for key, value in abs_i_tokens.items()}
        abs_q_tokens = {key: value.cuda() for key, value in abs_q_tokens.items()}

        v2_logits = self.model(v2_i_tokens, v2_q_tokens, alpha)
        abs_logits = self.model(
            abs_i_tokens, abs_q_tokens, alpha
        )

        v2_label, abs_label = v2_label.cuda(), abs_label.cuda()
        v2_label_type, abs_label_type = v2_label_type.cuda(), abs_label_type.cuda()

        v2_domain_label = self.v2_domain_label.repeat(v2_label.shape[0], 1).cuda()
        abs_domain_label = self.abs_domain_label.repeat(abs_label.shape[0], 1).cuda()
        
        if not self.cfg['use_label_type_classifier']:
            v2_label_logits, v2_domain_logits = v2_logits
            abs_label_logits, abs_domain_logits = abs_logits

            args = [
                v2_label_logits,
                v2_domain_logits,
                abs_label_logits,
                abs_domain_logits,
                v2_label,
                abs_label,
                v2_domain_label,
                abs_domain_label,
            ]
        else:
            v2_label_logits, v2_domain_logits, v2_label_type_logits = v2_logits
            abs_label_logits, abs_domain_logits, abs_label_type_logits = abs_logits

            args = [
                v2_label_logits,
                v2_domain_logits,
                abs_label_logits,
                abs_domain_logits,
                v2_label,
                abs_label,
                v2_domain_label,
                abs_domain_label,

                v2_label_type_logits,
                abs_label_type_logits,
                v2_label_type,
                abs_label_type,
            ]


        losses = self.get_loss(*args)

        accuracies = self.get_accuracy(*args)

        return (losses, accuracies)

    def train_epoch(self):
        self.model.train()
        label_running_loss = 0.0
        domain_running_loss = 0.0
        total_running_loss = 0.0

        for i, (
                (v2_i_tokens, v2_q_tokens, v2_label, v2_label_type),
                (abs_i_tokens, abs_q_tokens, abs_label, abs_label_type),
        ) in enumerate(self.train_dataloader):

            self.alpha = self.get_alpha(i, self.num_train_batches)

            (label_loss, domain_loss, label_type_loss, total_loss), _ = self.process_input(
                v2_i_tokens,
                v2_q_tokens,
                v2_label,
                abs_i_tokens,
                abs_q_tokens,
                abs_label,
                self.alpha,

                v2_label_type,
                abs_label_type,
            )

            label_running_loss += label_loss.item()
            domain_running_loss += domain_loss.item()
            total_running_loss += total_loss.item()

            if self.cfg['print_logs'] and self.num_train_batches > 4 and i % (self.num_train_batches // 4) == 0:
                print(
                    f"\t Iter [{i}/{self.num_train_batches}]\t Loss: {total_loss.item():.6f}"
                )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        avg_label_loss = label_running_loss / self.num_train_batches
        avg_domain_loss = domain_running_loss / self.num_train_batches
        avg_total_loss = total_running_loss / self.num_train_batches

        return avg_label_loss, avg_domain_loss, avg_total_loss

    def eval_epoch(self):
        v2_correct, v2_total = 0, 0
        abs_correct, abs_total = 0, 0
        correct, total = 0, 0
        domain_correct, domain_total = 0, 0

        running_loss = 0.0

        for i, (
                (v2_i_tokens, v2_q_tokens, v2_label),
                (
                        abs_i_tokens,
                        abs_q_tokens,
                        abs_label,
                ),
        ) in enumerate(self.val_dataloader):
            (
                (label_loss, domain_loss, total_loss),
                (
                    _v2_total,
                    _v2_correct,
                    _abs_total,
                    _abs_correct,
                    _total,
                    _correct,
                    _domain_total,
                    _domain_correct,
                ),
            ) = self.process_input(
                v2_i_tokens,
                v2_q_tokens,
                v2_label,
                abs_i_tokens,
                abs_q_tokens,
                abs_label,
                self.alpha,
            )

            v2_total += _v2_total
            v2_correct += _v2_correct
            abs_total += _abs_total
            abs_correct += _abs_correct
            total += _total
            correct += _correct
            domain_total += _domain_total
            domain_correct += _domain_correct

            running_loss += total_loss.item()

        eval_loss = running_loss / self.num_val_batches
        v2_accuracy = v2_correct / v2_total
        abs_accuracy = abs_correct / abs_total
        total_accuracy = correct / total

        domain_accuracy = domain_correct / domain_total

        return eval_loss, v2_accuracy, abs_accuracy, total_accuracy, domain_accuracy

    def train(self, show_plot, optuna_trial=None, comet_expt=None):
        min_eval_loss = float("inf")
        high_eval_loss_count = 0

        for self.epoch in range(self.num_epochs):
            self.train_dataloader = zip(
                self.v2_train_dataloader, self.abs_train_dataloader
            )
            self.val_dataloader = zip(self.v2_val_dataloader, self.abs_val_dataloader)

            label_loss, domain_loss, total_loss = self.train_epoch()
            with torch.no_grad():
                (
                    eval_loss,
                    v2_accuracy,
                    abs_accuracy,
                    total_accuracy,
                    domain_accuracy,
                ) = self.eval_epoch()

            self.scheduler.step()

            self.train_losses.append((label_loss, domain_loss, total_loss))
            self.eval_losses.append(eval_loss)
            self.accuracies.append(
                (v2_accuracy, abs_accuracy, total_accuracy, domain_accuracy)
            )

            # Plotting
            if show_plot and self.epoch > 0 and self.epoch % 10 == 0:
                self.plot(self.epoch + 1)
            
            # Logging
            if self.cfg['print_logs']:
                print(
                    f"Epoch [{self.epoch + 1}/{self.num_epochs}]\t \
                        Avg Train Loss: {total_loss:.6f}\t \
                        Avg Eval Loss: {eval_loss:.6f}\t \
                        Avg Domain Accuracy: {domain_accuracy:.2f}\t \
                        Avg Eval Accuracy: {total_accuracy:.2f}"
                )
            
            # Comet Logging
            if comet_expt:
                comet_expt.log_metrics({
                    'Loss/Train_label': label_loss,
                    'Loss/Train_domain': domain_loss,
                    'Loss/Train_total': total_loss,
                    'Loss/Eval': eval_loss,
                    'Accuracy/Domain': domain_accuracy,
                    'Accuracy/v2_label': v2_accuracy,
                    'Accuracy/abs_label': abs_accuracy,
                    'Accuracy/avg_label': total_accuracy
                }, step=self.epoch)

            
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


if __name__ == '__main__':
    cfg = {
        'name': 'DANN',

        ### DataLoader ###
        'n_classes': 10,
        'n_types': 6,

        'v2_samples_per_answer': 150,
        'abs_samples_per_answer': 150,
        'source_domain': 'v2',
        
        
        ### VLModel ###
        'image_encoder': 'facebook/dinov2-base',
        'text_encoder': 'bert-base-uncased',
        
        ## Embedder
        'num_attn_heads': 8,
        'fusion_mode': 'cat',
        'num_stacked_attn': 1, 
        
        'criss_cross__drop_p': 0.0,
        'post_concat__drop_p': 0.0, 
        'embed_attn__add_residual': False,
        'embed_attn__drop_p': 0.0,

        ## Label Classifier
        # 'use_label_type_classifier': False,
        # 'use_label_type_classifier': True,
        'num_label_types': 5,

        'label_classifier__use_bn': True,
        'label_classifier__drop_p': 0.0,
        'label_classifier__repeat_layers': [0, 0], 

        ## Domain Classifier
        'domain_classifier__use_bn': True,
        'domain_classifier__drop_p': 0.5,
        'domain_classifier__repeat_layers': [2, 2], 


        ### Objective ###
        'domain_adaptation_method': 'domain_adversarial',  # 'naive', 'importance_sampling', 'domain_adversarial'


        ### Trainer ###
        'relaxation_period': -1,

        'epochs': 30,
        'batch_size': 150,
        'base_lr': 0.0005,
        'weight_decay': 1e-5,
        
        ### Logging ###
        'print_logs': False,
        'show_plot': True,
        
        'weights_save_root': './weights/raw'
    }


    if True:
        # v2
        cfg['source_domain'] = 'v2'
        trainer = DA_Trainer(cfg, vqa_v2, vqa_abs)
        v2_ckpt_path = cfg['weights_save_path']

        trainer.train(show_plot=True)

        # abs
        cfg['source_domain'] = 'abs'
        trainer = DA_Trainer(cfg, vqa_v2, vqa_abs)
        abs_ckpt_path = cfg['weights_save_path']

        trainer.train(show_plot=False)