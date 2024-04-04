from pt_dataset import OracleDataset
from oracle_gen import get_trainable_image

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer


class OracleDetectionTrainer(DetectionTrainer):
    def __init__(self, train_dataloader, test_dataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.data = {"nc": None, "names": "Oracle"}
        self.model = "yolov8n.yaml"
    
    def get_dataset(self):
        return None, None

    def get_dataloader(self, mode="train", *args, **kwargs):
        if mode == "train":
            return self.train_dataloader
        elif mode == "test" or mode == "val":
            return self.train_dataloader


def main():
    train_dataset = OracleDataset(get_trainable_image, "00_Oracle/LOBI_Roots", batch_size=16, length=1024)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=6)
    test_dataset = OracleDataset(get_trainable_image, "00_Oracle/LOBI_Roots", batch_size=16, length=128)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)

    trainer = OracleDetectionTrainer(train_dataloader, test_dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
