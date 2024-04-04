from oracle_gen import get_trainable_image

import torch
from torch.utils.data import Dataset, DataLoader

class OracleDataset(Dataset):
    def __init__(self, generator, root_path, batch_size, length=1000):
        self.generator = generator
        self.root_path = root_path
        self.batch_size = batch_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        n_roots_oracle = (idx // self.batch_size) % 5 + 1
        img, bbox, label = self.generator(self.root_path, n_roots=n_roots_oracle)
        # Assuming your generator handles single items per __getitem__
        # You may need to adjust depending on your generator's output
        return {"img": torch.tensor(img), "bboxes": torch.tensor(bbox), "labels": torch.tensor(label)}


def main():
    # Create a PyTorch dataset and dataloader
    dataset = OracleDataset(get_trainable_image, "00_Oracle/LOBI_Roots", batch_size=16, length=1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)

    # Iterate over the dataloader
    for batch in dataloader:
        img, bbox, label = batch
        print(bbox, label)


if __name__ == "__main__":
    main()
