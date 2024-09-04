import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import VQADataset, data_processing_v2, np, torch
from models.VLModel import VLModel
from models.models import VLModel, nn

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
        # for k in [
        #     "name",
        #     "n_classes",
        #     "v2_samples_per_answer",
        #     "abs_samples_per_answer",
        #     "source_domain",
        #     "base_lr",
        #     # "domain_adaptation_method",
        # ]:
        #     v = self.cfg[k]
        #     if k != "base_lr":
        #         title += f"{k}={v}__"
        #     else:
        #         title += f"{k}={v:.2e}__"
        title = title.replace(" ", "_")

        def random_string(n):
            chars = np.array(
                list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            )
            return "".join(np.random.choice(chars, n))

        self.cfg["title"] = title + random_string(8)

        self.cfg["weights_save_path"] = (
            self.cfg["weights_save_root"] + "/" + self.cfg["title"] + ".pth"
        )

        if cfg["print_logs"]:
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

            if (
                self.cfg["print_logs"]
                and num_batches > 16
                and i % (num_batches // 4) == 0
            ):
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