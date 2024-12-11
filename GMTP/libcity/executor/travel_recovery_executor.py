import os
import time
import numpy as np
import torch
from libcity.executor.bert_executor import BertBaseExecutor
from libcity.model import loss
from libcity.model.trajectory_embedding import TrajectoryRecoveryModel
from tqdm import tqdm


class TrajectoryRecoveryExecutor(object):
    def __init__(self, config, model, data_feature):
        self.config = config
        self.model = model
        self.data_feature = data_feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self._logger = getLogger('TrajectoryRecoveryExecutor')
        self._logger.setLevel('INFO')

        self.pretrain_model_path = self.config.get("pretrain_path", None)
        if self.pretrain_model_path:
            self._load_pretrain_model()

    def _load_pretrain_model(self):
        checkpoint = torch.load(self.pretrain_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info(f'Loaded pretrained model from {self.pretrain_model_path}')

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        self.model.train()
        best_loss = float('inf')

        for epoch in range(self.config.get('epochs', 100)):
            train_loss = self._train_epoch(train_dataloader, epoch)
            val_loss = self._valid_epoch(eval_dataloader, epoch, mode='Val')

            self.scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                self._logger.info(f'Validation loss improved from {best_loss} to {val_loss}')

            if test_dataloader is not None:
                self._valid_epoch(test_dataloader, epoch, mode='Test')

    def _train_epoch(self, train_dataloader, epoch):
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{self.config.get("epochs", 100)}')):
            X, targets = batch  # 假设batch只包含特征和目标
            X, targets = X.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * targets.size(0)

        avg_loss = total_loss / len(train_dataloader.dataset)
        self._logger.info(f'Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}')
        return avg_loss

    def _valid_epoch(self, eval_dataloader, epoch, mode):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader, desc=f'{mode} Epoch {epoch + 1}')):
                X, targets = batch
                X, targets = X.to(self.device), targets.to(self.device)

                outputs = self.model(X)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * targets.size(0)

        avg_loss = total_loss / len(eval_dataloader.dataset)
        self._logger.info(f'{mode} Epoch {epoch + 1}, {mode} Loss: {avg_loss:.4f}')
        return avg_loss

    # 使用示例（假设config, train_dataloader, eval_dataloader等已定义）
# config = {...}  # 您的配置字典
# model = TrajectoryRecoveryModel(d_model=768)  # 假设您的BERT输出维度为768
# train_dataloader, eval_dataloader = ...  # 您的数据加载器
# executor = TrajectoryRecoveryExecutor(config, model, None)  # data_feature在这里可能未使用，但您可以根据需要传递它
# executor.train(train_dataloader, eval_dataloader)