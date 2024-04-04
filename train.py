from pt_dataset import OracleDataset
from oracle_gen import get_trainable_image

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer


class OracleDetectionTrainer(DetectionTrainer):
    def __init__(self, dataloader, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.dataloader = dataloader

    def get_dataloader(self, cfg, weights):
        return self.dataloader


def main():
    dataset = OracleDataset(get_trainable_image, "00_Oracle/LOBI_Roots", batch_size=16, length=1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)
    trainer = OracleDetectionTrainer(dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
