from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    transfer_weights,
    show_series,
    show_matrix,
)
from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class Exp_HtulTS(Exp_Basic):
    def __init__(self, args):
        super(Exp_HtulTS, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        if self.args.downstream_task == "forecast":
            model = self.model_dict[self.args.model].Model(self.args).float()
        elif self.args.downstream_task == "classification":
            model = self.model_dict[self.args.model].ClsModel(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            # Load to CPU first to handle checkpoints saved on different GPU devices
            state_dict = torch.load(self.args.load_checkpoints, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)  # strict=False allows missing keys (e.g., head)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print(
            "number of model params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.task_name == "finetune" and self.args.downstream_task == "classification":
            criterion = nn.CrossEntropyLoss()
            print("Using CrossEntropyLoss")
        else:
            criterion = nn.MSELoss()
            print("Using MSELoss")
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        # model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,T_max=self.args.train_epochs)
        model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model_optim, gamma=self.args.lr_decay
        )

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # current learning rate
            print("Current learning rate: {:.7f}".format(model_scheduler.get_last_lr()[0]))

            train_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            vali_loss = self.valid_one_epoch(vali_loader)

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {}/{}, Time: {:.2f}, Train Loss: {:.4f}, Vali Loss: {:.4f}".format(
                    epoch + 1,
                    self.args.train_epochs,
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                )
            )

            loss_scalar_dict = {
                "train_loss": train_loss,
                "vali_loss": vali_loss,
            }

            self.writer.add_scalars(f"pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model epoch{}...".format(
                        min_vali_loss, vali_loss, epoch
                    )
                )
                min_vali_loss = vali_loss
                # Save the full model state dict
                torch.save(self.model.state_dict(), os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))
                torch.save(self.model.state_dict(), os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        train_loss = []
        model_criterion = self._select_criterion()

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            train_loader
        ):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            pred_x = self.model(batch_x)
            diff_loss = model_criterion(pred_x, batch_x)
            diff_loss.backward()

            model_optim.step()
            train_loss.append(diff_loss.item())

        model_scheduler.step()
        train_loss = np.mean(train_loss)

        return train_loss

    def valid_one_epoch(self, vali_loader):
        vali_loss = []
        model_criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)
                diff_loss = model_criterion(pred_x, batch_x)
                vali_loss.append(diff_loss.item())

        vali_loss = np.mean(vali_loss)

        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loader = tqdm(train_loader, desc="Training")

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                loss = model_criteria(pred_x, batch_y)
                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.mean(train_loss)
            vali_loss = self.valid(vali_loader, model_criteria)
            test_loss = self.valid(test_loader, model_criteria)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1,
                    len(train_loader),
                    end_time - start_time,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}\n".format(
                        epoch + 1,
                        len(train_loader),
                        end_time - start_time,
                        train_loss,
                        vali_loss,
                        test_loss,
                    )
                )
            early_stopping(vali_loss, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))

        self.lr = model_scheduler.get_last_lr()[0]

        return self.model

    def valid(self, vali_loader, model_criteria):
        vali_loss = []
        self.model.eval()
        vali_loader = tqdm(vali_loader, desc="Validation")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                loss = model_criteria(pred_x, batch_y)
                vali_loss.append(loss.item())

        vali_loss = np.mean(vali_loss)
        self.model.train()

        return vali_loss

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        preds = []
        trues = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred_x = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0

                pred_x = pred_x[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                pred = pred_x.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, _, _, _ = metric(preds, trues)
        print(
            "{0}->{1}, mse:{2:.3f}, mae:{3:.3f}".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f = open(folder_path + "/score.txt", "a")
        f.write(
            "{0}->{1}, {2:.3f}, {3:.3f} \n".format(
                self.args.input_len, self.args.pred_len, mse, mae
            )
        )
        f.close()

    def cls_train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_criteria = self._select_criterion()
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            train_loss = []
            train_acc = []
            train_f1 = []

            print("Current learning rate: {:.7f}".format(model_optim.param_groups[0]['lr']))

            self.model.train()
            train_loader = tqdm(train_loader)
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                loss.backward()
                model_optim.step()
                if self.args.lradj == "step":
                    adjust_learning_rate(
                        model_optim,
                        model_scheduler,
                        epoch + 1,
                        self.args,
                        printout=False,
                    )
                    model_scheduler.step()

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')

                train_loss.append(loss.item())
                train_acc.append(acc)
                train_f1.append(f1)

            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            train_f1 = np.mean(train_f1)

            vali_loss, vali_acc, vali_f1 = self.cls_valid(vali_loader, model_criteria)
            test_loss, test_acc, test_f1 = self.cls_valid(test_loader, model_criteria)
            self.cls_test(write_log=False)

            end_time = time.time()
            print(
                "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | ".format(
                    epoch + 1, len(train_loader), end_time - start_time
                ) +
                "Train Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    train_loss, train_acc, train_f1
                ) +
                "Vali Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f} | ".format(
                    vali_loss, vali_acc, vali_f1
                ) +
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                    test_loss, test_acc, test_f1
                )
            )
            log_path = path + "/" + "log.txt"
            with open(log_path, "a") as log_file:
                log_file.write(
                    "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f}, Acc: {4:.4f}, F1: {5:.4f} | Vali Loss: {6:.7f}, Acc: {7:.4f}, F1: {8:.4f} | Test Loss: {9:.7f}, Acc: {10:.4f}, F1: {11:.4f}\n".format(
                        epoch + 1, len(train_loader), end_time - start_time, train_loss, train_acc, train_f1, vali_loss, vali_acc, vali_f1, test_loss, test_acc, test_f1
                    )
                )

            early_stopping(-vali_acc, self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        return self.model

    def cls_valid(self, vali_loader, model_criteria):
        vali_acc = []
        vali_f1 = []
        vali_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                vali_loss.append(loss.item())
                acc = accuracy_score(trues, preds)
                f1 = f1_score(trues, preds, average='macro')
                vali_acc.append(acc)
                vali_f1.append(f1)
        
        vali_loss = np.mean(vali_loss)
        vali_acc = np.mean(vali_acc)
        vali_f1 = np.mean(vali_f1)

        return vali_loss, vali_acc, vali_f1

    def cls_test(self, write_log=True):
        test_data, test_loader = self._get_data(flag="test")
        model_criteria = self._select_criterion()

        preds_all = []
        trues_all = []
        test_loss = []

        folder_path = "./outputs/test_results/{}".format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                outputs = self.model(batch_x)
                loss = model_criteria(outputs, batch_y)

                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()

                test_loss.append(loss.item())
                preds_all.extend(preds)
                trues_all.extend(trues)

        test_loss = np.mean(test_loss)
        test_acc = accuracy_score(trues_all, preds_all)
        test_f1 = f1_score(trues_all, preds_all, average='macro')

        print(
            "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}".format(
                test_loss, test_acc, test_f1
            )
        )
        if write_log:
            f = open(folder_path + "/score.txt", "a")
            f.write(
                "Test Loss: {:.7f}, Acc: {:.4f}, F1: {:.4f}\n".format(test_loss, test_acc, test_f1)
            )
            f.close()

    # ==========================================================================
    # MULTI-TASK TRAINING METHODS
    # ==========================================================================
    
    def multi_task_train(self, setting):
        """
        Multi-task joint training for classification with reconstruction.
        
        For classification-only datasets (like UCR, Epilepsy, HAR):
        - Primary task: Classification (CrossEntropy)
        - Auxiliary task: Reconstruction/Denoising (MSE on input)
        
        This aligns pretraining and finetuning objectives.
        """
        # Get data (uses current downstream_task setting)
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        
        # Determine mode based on downstream_task
        is_classification = self.args.downstream_task == "classification"
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Setup
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.01)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        recon_criterion = nn.MSELoss()  # For reconstruction/forecasting
        cls_criterion = nn.CrossEntropyLoss()
        
        model_scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )
        
        # Multi-task parameters
        multi_task_lambda = getattr(self.args, 'multi_task_lambda', 0.3)
        gradient_clip = getattr(self.args, 'gradient_clip', 1.0)
        multi_task_schedule = getattr(self.args, 'multi_task_schedule', 'warmup')
        
        best_recon_mse = float('inf')
        best_cls_acc = 0.0
        
        print(f"\n{'='*60}")
        print(f"Multi-Task Training: Classification + Reconstruction")
        print(f"Lambda (classification weight): {multi_task_lambda}")
        print(f"Schedule: {multi_task_schedule}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.args.train_epochs):
            start_time = time.time()
            
            # Calculate lambda for this epoch based on schedule
            if multi_task_schedule == 'warmup':
                warmup_epochs = int(0.2 * self.args.train_epochs)
                if epoch < warmup_epochs:
                    current_lambda = multi_task_lambda * (epoch / max(warmup_epochs, 1))
                else:
                    current_lambda = multi_task_lambda
            elif multi_task_schedule == 'linear':
                current_lambda = multi_task_lambda * (epoch / max(self.args.train_epochs - 1, 1))
            elif multi_task_schedule == 'cosine':
                progress = epoch / max(self.args.train_epochs - 1, 1)
                current_lambda = multi_task_lambda * (1 - np.cos(progress * np.pi)) / 2
            else:
                current_lambda = multi_task_lambda
            
            print(f"\nEpoch {epoch+1}/{self.args.train_epochs}, Lambda: {current_lambda:.4f}")
            print(f"Current LR: {model_optim.param_groups[0]['lr']:.7f}")
            
            # Training epoch - uses same data loader for both tasks
            train_metrics = self._multi_task_cls_train_epoch(
                train_loader,
                model_optim, model_scheduler,
                recon_criterion, cls_criterion,
                current_lambda, gradient_clip
            )
            
            # Validation
            vali_metrics = self._multi_task_cls_validate(
                vali_loader,
                recon_criterion, cls_criterion,
                current_lambda
            )
            
            # Test
            test_metrics = self._multi_task_cls_validate(
                test_loader,
                recon_criterion, cls_criterion,
                current_lambda
            )
            
            end_time = time.time()
            
            print(
                f"Time: {end_time-start_time:.2f}s | "
                f"Train: Loss={train_metrics['total_loss']:.4f}, Recon={train_metrics['recon_loss']:.4f}, "
                f"Cls={train_metrics['cls_loss']:.4f}, Acc={train_metrics['cls_acc']:.4f} | "
                f"Vali: Recon={vali_metrics['recon_loss']:.4f}, Acc={vali_metrics['cls_acc']:.4f} | "
                f"Test: Recon={test_metrics['recon_loss']:.4f}, Acc={test_metrics['cls_acc']:.4f}"
            )
            
            # Log
            self.writer.add_scalars('multi_task/train', {
                'total_loss': train_metrics['total_loss'],
                'recon_loss': train_metrics['recon_loss'],
                'cls_loss': train_metrics['cls_loss'],
            }, epoch)
            self.writer.add_scalars('multi_task/vali', {
                'recon_loss': vali_metrics['recon_loss'],
                'cls_acc': vali_metrics['cls_acc'],
            }, epoch)
            
            # Track best metrics
            if vali_metrics['recon_loss'] < best_recon_mse:
                best_recon_mse = vali_metrics['recon_loss']
            if vali_metrics['cls_acc'] > best_cls_acc:
                best_cls_acc = vali_metrics['cls_acc']
            
            # Early stopping on classification accuracy (primary task)
            # Use negative accuracy so lower is better for early stopping
            early_stopping(-vali_metrics['cls_acc'], self.model, path=path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj != "step":
                adjust_learning_rate(model_optim, model_scheduler, epoch + 1, self.args)
        
        # Load best model
        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
        
        print(f"\nBest Recon MSE: {best_recon_mse:.4f}, Best Cls Acc: {best_cls_acc:.4f}")
        
        return self.model
    
    def _multi_task_cls_train_epoch(self, data_loader, optimizer, scheduler,
                                     recon_criterion, cls_criterion, lambda_cls, gradient_clip):
        """
        Train one epoch with multi-task objective for classification datasets.
        Uses the SAME data for both classification and reconstruction tasks.
        
        Loss = (1 - λ) * MSE_reconstruction + λ * CrossEntropy_classification
        """
        self.model.train()
        
        metrics = {
            'total_loss': [],
            'recon_loss': [],
            'cls_loss': [],
            'cls_acc': [],
        }
        
        pbar = tqdm(data_loader, desc="Training")
        
        for batch_x, batch_y, _, _ in pbar:
            optimizer.zero_grad()
            
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.long().to(self.device)  # Classification labels
            
            # Add noise for reconstruction task (denoising objective)
            noise_level = getattr(self.args, 'noise_level', 0.15)
            if noise_level > 0:
                noise = torch.randn_like(batch_x) * noise_level * batch_x.std()
                noisy_x = batch_x + noise
            else:
                noisy_x = batch_x
            
            # Forward pass - get both reconstruction and classification outputs
            # The model should reconstruct the clean input from noisy input
            # and also classify the input
            
            # Reconstruction: predict clean input from noisy input
            if hasattr(self.model, 'forecast_forward'):
                recon_pred = self.model.forecast_forward(noisy_x)
            elif hasattr(self.model, 'forward_pretrain'):
                # Use pretrain forward which includes denoising
                recon_pred, _ = self.model.forward_pretrain(noisy_x)
            else:
                recon_pred = self.model(noisy_x)
            
            # Match dimensions for reconstruction loss
            if recon_pred.shape != batch_x.shape:
                # Adjust to match input shape
                min_len = min(recon_pred.shape[1], batch_x.shape[1])
                recon_pred = recon_pred[:, :min_len, :]
                target_x = batch_x[:, :min_len, :]
            else:
                target_x = batch_x
            
            recon_loss = recon_criterion(recon_pred, target_x)
            
            # Classification forward
            if hasattr(self.model, 'classify_forward'):
                cls_logits = self.model.classify_forward(batch_x)
            elif hasattr(self.model, 'forward_cls'):
                cls_logits = self.model.forward_cls(batch_x)
            else:
                # Fallback: use the model directly
                cls_logits = self.model(batch_x)
            
            cls_loss = cls_criterion(cls_logits, batch_y)
            
            preds = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
            trues = batch_y.detach().cpu().numpy()
            cls_acc = accuracy_score(trues, preds)
            
            # Compute total loss
            # Loss = (1 - λ) * MSE_reconstruction + λ * CrossEntropy_classification
            total_loss = (1 - lambda_cls) * recon_loss + lambda_cls * cls_loss
            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
            
            if self.args.lradj == "step":
                scheduler.step()
            
            metrics['total_loss'].append(total_loss.item())
            metrics['recon_loss'].append(recon_loss.item())
            metrics['cls_loss'].append(cls_loss.item())
            metrics['cls_acc'].append(cls_acc)
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'acc': f"{cls_acc:.4f}"
            })
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _multi_task_cls_validate(self, data_loader, recon_criterion, cls_criterion, lambda_cls):
        """Validate on both reconstruction and classification tasks"""
        self.model.eval()
        
        metrics = {
            'recon_loss': [],
            'cls_loss': [],
            'cls_preds': [],
            'cls_trues': [],
        }
        
        with torch.no_grad():
            for batch_x, batch_y, _, _ in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                
                # Reconstruction (without noise for validation)
                if hasattr(self.model, 'forecast_forward'):
                    recon_pred = self.model.forecast_forward(batch_x)
                elif hasattr(self.model, 'forward_pretrain'):
                    recon_pred, _ = self.model.forward_pretrain(batch_x)
                else:
                    recon_pred = self.model(batch_x)
                
                # Match dimensions
                if recon_pred.shape != batch_x.shape:
                    min_len = min(recon_pred.shape[1], batch_x.shape[1])
                    recon_pred = recon_pred[:, :min_len, :]
                    target_x = batch_x[:, :min_len, :]
                else:
                    target_x = batch_x
                
                recon_loss = recon_criterion(recon_pred, target_x)
                metrics['recon_loss'].append(recon_loss.item())
                
                # Classification
                if hasattr(self.model, 'classify_forward'):
                    cls_logits = self.model.classify_forward(batch_x)
                elif hasattr(self.model, 'forward_cls'):
                    cls_logits = self.model.forward_cls(batch_x)
                else:
                    cls_logits = self.model(batch_x)
                
                cls_loss = cls_criterion(cls_logits, batch_y)
                metrics['cls_loss'].append(cls_loss.item())
                
                preds = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
                trues = batch_y.detach().cpu().numpy()
                metrics['cls_preds'].extend(preds)
                metrics['cls_trues'].extend(trues)
        
        result = {
            'recon_loss': np.mean(metrics['recon_loss']),
            'cls_loss': np.mean(metrics['cls_loss']),
            'cls_acc': accuracy_score(metrics['cls_trues'], metrics['cls_preds']),
        }
        
        result['total_loss'] = (1 - lambda_cls) * result['recon_loss'] + lambda_cls * result['cls_loss']
        
        return result
    
    # Keep old methods for backward compatibility with joint forecast+classification
    def _multi_task_train_epoch_legacy(self, forecast_loader, cls_loader, optimizer, scheduler,
                                 forecast_criterion, cls_criterion, lambda_cls, gradient_clip):
        """Legacy: Train one epoch with separate forecast and classification data"""
        self.model.train()
        
        metrics = {
            'total_loss': [],
            'forecast_loss': [],
            'cls_loss': [],
            'cls_acc': [],
        }
        
        forecast_iter = iter(forecast_loader)
        cls_iter = iter(cls_loader) if cls_loader is not None else None
        
        n_iters = len(forecast_loader)
        pbar = tqdm(range(n_iters), desc="Training")
        
        for i in pbar:
            optimizer.zero_grad()
            
            # Get forecast batch
            try:
                f_batch_x, f_batch_y, _, _ = next(forecast_iter)
            except StopIteration:
                forecast_iter = iter(forecast_loader)
                f_batch_x, f_batch_y, _, _ = next(forecast_iter)
            
            f_batch_x = f_batch_x.float().to(self.device)
            f_batch_y = f_batch_y.float().to(self.device)
            
            # Forecast forward
            if hasattr(self.model, 'forecast_forward'):
                forecast_pred = self.model.forecast_forward(f_batch_x)
            else:
                forecast_pred = self.model(f_batch_x)
            
            f_dim = -1 if self.args.features == "MS" else 0
            forecast_pred = forecast_pred[:, -self.args.pred_len:, f_dim:]
            f_batch_y = f_batch_y[:, -self.args.pred_len:, f_dim:]
            
            forecast_loss = forecast_criterion(forecast_pred, f_batch_y)
            
            # Classification forward (if data available)
            cls_loss = torch.tensor(0.0, device=self.device)
            cls_acc = 0.0
            
            if cls_iter is not None:
                try:
                    c_batch_x, c_batch_y, _, _ = next(cls_iter)
                except StopIteration:
                    cls_iter = iter(cls_loader)
                    c_batch_x, c_batch_y, _, _ = next(cls_iter)
                
                c_batch_x = c_batch_x.float().to(self.device)
                c_batch_y = c_batch_y.long().to(self.device)
                
                if hasattr(self.model, 'classify_forward'):
                    cls_logits = self.model.classify_forward(c_batch_x)
                else:
                    cls_logits = self.model(c_batch_x)
                
                cls_loss = cls_criterion(cls_logits, c_batch_y)
                
                preds = torch.argmax(cls_logits, dim=1).detach().cpu().numpy()
                trues = c_batch_y.detach().cpu().numpy()
                cls_acc = accuracy_score(trues, preds)
            
            # Compute total loss
            total_loss = (1 - lambda_cls) * forecast_loss + lambda_cls * cls_loss
            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            optimizer.step()
            
            if self.args.lradj == "step":
                scheduler.step()
            
            metrics['total_loss'].append(total_loss.item())
            metrics['forecast_loss'].append(forecast_loss.item())
            metrics['cls_loss'].append(cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss)
            metrics['cls_acc'].append(cls_acc)
            
            pbar.set_postfix({
                'loss': total_loss.item(),
                'f_mse': forecast_loss.item(),
                'c_acc': cls_acc
            })
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _multi_task_validate_legacy(self, forecast_loader, cls_loader,
                              forecast_criterion, cls_criterion, lambda_cls):
        """Legacy: Validate on both tasks with separate data loaders"""
        self.model.eval()
        
        metrics = {
            'forecast_loss': [],
            'cls_loss': [],
            'cls_preds': [],
            'cls_trues': [],
        }
        
        with torch.no_grad():
            # Forecast validation
            for batch_x, batch_y, _, _ in forecast_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if hasattr(self.model, 'forecast_forward'):
                    pred = self.model.forecast_forward(batch_x)
                else:
                    pred = self.model(batch_x)
                
                f_dim = -1 if self.args.features == "MS" else 0
                pred = pred[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                loss = forecast_criterion(pred, batch_y)
                metrics['forecast_loss'].append(loss.item())
            
            # Classification validation
            if cls_loader is not None:
                for batch_x, batch_y, _, _ in cls_loader:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.long().to(self.device)
                    
                    if hasattr(self.model, 'classify_forward'):
                        logits = self.model.classify_forward(batch_x)
                    else:
                        logits = self.model(batch_x)
                    
                    loss = cls_criterion(logits, batch_y)
                    metrics['cls_loss'].append(loss.item())
                    
                    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    trues = batch_y.detach().cpu().numpy()
                    metrics['cls_preds'].extend(preds)
                    metrics['cls_trues'].extend(trues)
        
        result = {
            'forecast_loss': np.mean(metrics['forecast_loss']),
            'cls_loss': np.mean(metrics['cls_loss']) if metrics['cls_loss'] else 0.0,
            'cls_acc': accuracy_score(metrics['cls_trues'], metrics['cls_preds']) if metrics['cls_trues'] else 0.0,
        }
        
        result['total_loss'] = (1 - lambda_cls) * result['forecast_loss'] + lambda_cls * result['cls_loss']
        
        return result
    
    def multi_task_test(self):
        """Test classification performance after multi-task training"""
        # For classification-only multi-task, just run cls_test
        if self.args.downstream_task == "classification":
            print("\n=== Classification Test (Multi-Task) ===")
            self.cls_test()
        else:
            # For forecast+classification multi-task (legacy)
            print("\n=== Forecasting Test ===")
            self.test()
            
            try:
                original_task = self.args.downstream_task
                self.args.downstream_task = "classification"
                print("\n=== Classification Test ===")
                self.cls_test()
                self.args.downstream_task = original_task
            except Exception as e:
                print(f"Could not test classification: {e}")