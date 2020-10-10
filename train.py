from Backbone import DAL_model
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch import autograd
import os
from custom_datasets import ImageFolderWithAges
from meta import age_cutoffs
from utils import Recorder
from itertools import chain
from albumentations.pytorch.transforms import ToTensor
from albumentations import (
    Compose, HorizontalFlip, Resize, OneOf, IAAAdditiveGaussianNoise, GaussNoise
)
import cv2
import numpy as np
from torch.cuda.amp import GradScaler, autocast

from pytorch_helper_bot import (
    MultiStageScheduler, LinearLR,
    CosineAnnealingLR
)
from pytorch_helper_bot.optimizers import RAdam
from pytorch_helper_bot.discriminative_learning_rates import set_trainable


def count_model_parameters(model):
    print(
        "# of parameters: {:,d}".format(
            np.sum(list(p.numel() for p in model.parameters()))))
    print(
        "# of trainable parameters: {:,d}".format(
            np.sum(list(p.numel() for p in model.parameters() if p.requires_grad))))


class Trainer():
    def __init__(
        self, model, dataset, ctx=-1, batch_size=128, optimizer='sgd', 
        grad_accu=1, lambdas=[0.05, 0.1], print_freq=32, train_head_only=True
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.finetune_layers = (
            # self.model.backbone.repeat_3[-1:], 
            self.model.backbone.last_bn, 
            self.model.backbone.last_linear, self.model.backbone.block8
        )        
        first_group = [
            {
                "params": chain(
                    self.model.age_classifier.parameters(),
                    self.model.RFM.parameters(),
                    self.model.margin_fc.parameters(),
                ),
                "lr": 5e-4
            }
        ]
        if not train_head_only:
            # first_group[0]["lr"] = 1e-4
            first_group.append(
                {
                    "params": chain(
                        *(x.parameters() for x in self.finetune_layers)
                    ),
                    "lr": 5e-5
                }
            )
        self.optbb = RAdam(first_group)
        self.optDAL = RAdam(self.model.DAL.parameters(), lr=5e-4)
        self.lambdas = lambdas
        self.print_freq = print_freq
        self.id_recorder = Recorder()
        self.age_recorder = Recorder()
        self.trainingDAL = False
        if ctx < 0:
            self.ctx = torch.device('cpu')
        else:
            self.ctx = torch.device(f'cuda:{ctx}')
        self.scaler1 = GradScaler()
        self.scaler2 = GradScaler()
        self.grad_accu = grad_accu
        self.train_head_only = train_head_only

    def train(self, epochs, start_epoch, save_path=None):
        self.train_ds = ImageFolderWithAges(
            self.dataset['pat'], self.dataset['pos'],
            transforms=Compose(
                [
                    HorizontalFlip(p=0.5),
                    OneOf([
                        IAAAdditiveGaussianNoise(),
                        GaussNoise(),
                    ], p=0.25),
                    Resize(200, 200, cv2.INTER_AREA),
                    ToTensor(normalize=dict(
                        mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
                    )
                ]
            ),
            root=self.dataset['train_root'],
        )
        self.train_ld = DataLoader(
            self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=2, drop_last=True, pin_memory=True
        )
        print("# Batches:", len(self.train_ld))
        if self.dataset['val_root'] is not None:
            self.val_ds = ImageFolderWithAges(
                self.dataset['pat'], self.dataset['pos'],
                root=self.dataset['val_root'],
                transforms=Compose([
                    Resize(200, 200, cv2.INTER_AREA),
                    ToTensor(normalize=dict(
                        mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
                    )
                ])
            )
            self.val_ld = DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size,
                                     pin_memory=True, num_workers=1)
        self.model = self.model.to(self.ctx)
        total_steps = len(self.train_ld) * epochs
        lr_durations = [
            int(total_steps*0.05),
            int(np.ceil(total_steps*0.95))
        ]
        break_points = [0] + list(np.cumsum(lr_durations))[:-1]
        self.schedulers = [
            MultiStageScheduler(
                [
                    LinearLR(self.optbb, 0.01, lr_durations[0]),
                    CosineAnnealingLR(self.optbb, lr_durations[1], eta_min=1e-6)
                ],
                start_at_epochs=break_points
            ),
            MultiStageScheduler(
                [
                    LinearLR(self.optDAL, 0.01, lr_durations[0]),
                    CosineAnnealingLR(self.optDAL, lr_durations[1], eta_min=1e-6)
                ],
                start_at_epochs=break_points
            )
        ]
        if self.train_head_only:
            set_trainable(self.model.backbone, False)
            for module in self.model.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.track_running_stats = False
        else:
            set_trainable(self.model.backbone, False)
            # for module in self.model.backbone.modules():
            #     if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            #         module.track_running_stats = False
            for module in self.finetune_layers:
                set_trainable(module, True)
                # for submodule in chain([module], module.modules()):
                #     if isinstance(submodule, (nn.BatchNorm2d, nn.BatchNorm1d)):
                #         submodule.track_running_stats = True
        count_model_parameters(self.model)                
        # print(self.optbb.param_groups[-1]["lr"])
        # print(self.optDAL.param_groups[-1]["lr"])
        for epoch in range(epochs):
            print(f'---- epoch {epoch} ----')
            self.update()
            if self.dataset['val_root'] is not None:
                acc = self.validate()
            else:
                acc = -1.
            if save_path is not None:
                torch.save(self.model.state_dict(), os.path.join(save_path, f'{start_epoch+epoch}_{acc:.4f}.pth'))

    def update(self):
        print('    -- Training --')
        self.model.train()
        self.model.backbone.eval()
        if not self.train_head_only:
            for module in self.finetune_layers:
                module.train()
                # for submodule in chain([module], module.modules()):
                #     if isinstance(submodule, (nn.BatchNorm2d, nn.BatchNorm1d)):
                #         submodule.eval()
        self.id_recorder.reset()
        self.age_recorder.reset()
        for i, (xs, ys, agegrps) in enumerate(self.train_ld):
            if i % 80 == 0:  # canonical maximization procesure
                self.set_train_mode(False)
            elif i % 80 == 28:  # RFM optimize procesure
                self.set_train_mode(True)
            xs, ys, agegrps = xs.to(self.ctx), ys.to(self.ctx), agegrps.to(self.ctx)
            with autocast():
                self.model(xs, ys, agegrps=agegrps)
            idLoss, id_acc, ageLoss, age_acc, cc = self.model(xs, ys, agegrps=agegrps)
            #print(f'        ---\n{idLoss}\n{id_acc}\n{ageLoss}\n{age_acc}\n{cc}')
            total_loss = idLoss + ageLoss*self.lambdas[0] + cc*self.lambdas[1]
            total_loss /= self.grad_accu
            self.id_recorder.gulp(len(agegrps), idLoss.detach().item(), id_acc.detach().item())
            self.age_recorder.gulp(len(agegrps), ageLoss.detach().item(), age_acc.detach().item())
            if i % self.print_freq == 0:
                print(
                    f'        iter: {i} {i%70} total loss: {total_loss.item():.4f} ({idLoss.item():.4f}, {id_acc.item():.4f}, {ageLoss.item():.4f}, {age_acc.item():.4f}, {cc.item():.8f}) {self.optbb.param_groups[-1]["lr"]:.6f}')
            if self.trainingDAL:
                self.scaler1.scale(-1 * cc*self.lambdas[1]).backward()
                # total_loss.backward()
                # Trainer.flip_grads(self.model.DAL)
                if (i + 1) % self.grad_accu == 0:
                    # self.optDAL.step()
                    self.scaler1.step(self.optDAL)
                    self.scaler1.update()
                    self.optDAL.zero_grad()
            else:
                self.scaler2.scale(total_loss).backward()
                # total_loss.backward()
                # self.optbb.step()
                if (i + 1) % self.grad_accu == 0:
                    self.scaler2.step(self.optbb)
                    self.scaler2.update()
                    self.optbb.zero_grad()                    
            for scheduler in self.schedulers:
                scheduler.step()
        # show average training meta after epoch
        print(f'        {self.id_recorder.excrete().result_as_string()}')
        print(f'        {self.age_recorder.excrete().result_as_string()}')

    def validate(self):
        print('    -- Validating --')
        self.model.eval()
        self.id_recorder.reset()
        self.age_recorder.reset()
        for i, (xs, ys, agegrps) in enumerate(self.val_ld):
            xs, ys, agegrps = xs.to(self.ctx), ys.to(self.ctx), agegrps.to(self.ctx)
            with torch.no_grad():
                with autocast():
                    idLoss, id_acc, ageLoss, age_acc, cc = self.model(xs, ys, agegrps)
                # total_loss = idLoss + ageLoss*self.lambdas[0] + cc*self.lambdas[1]
                self.id_recorder.gulp(len(agegrps), idLoss.item(), id_acc.item())
                self.age_recorder.gulp(len(agegrps), ageLoss.item(), age_acc.item())
        # show average validation meta after epoch
        print(f'        {self.id_recorder.excrete().result_as_string()}')
        print(f'        {self.age_recorder.excrete().result_as_string()}')
        return self.id_recorder.acc

    def set_train_mode(self, state):
        self.trainingDAL = not state
    #     Trainer.set_grads(self.model.RFM, state)
    #     # Trainer.set_grads(self.model.backbone, state)
    #     Trainer.set_grads(self.model.margin_fc, state)
    #     Trainer.set_grads(self.model.age_classifier, state)
    #     Trainer.set_grads(self.model.DAL, not state)

    @staticmethod
    def set_grads(mod, state):
        for para in mod.parameters():
            para.requires_grad = state

    @staticmethod
    def flip_grads(mod):
        for para in mod.parameters():
            if para.requires_grad:
                para.grad = - para.grad
