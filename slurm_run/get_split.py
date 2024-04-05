import os
import random

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.01
TEST_SPLIT = 0.1

DATASET_DIR = "OracleDS"

def get_split(data_dir):
    """
    Split data into train, validation and test sets.
    """
    data = os.listdir(data_dir)
    n = len(data)
    random.shuffle(data)
    train_data = data[:int(n*TRAIN_SPLIT)]
    val_data = data[int(n*TRAIN_SPLIT): int(n*(TRAIN_SPLIT+VAL_SPLIT))]
    test_data = data[int(n*(TRAIN_SPLIT+VAL_SPLIT)): int(n*(TRAIN_SPLIT+VAL_SPLIT+TEST_SPLIT))]
    return train_data, val_data, test_data

def main():
    train_data, val_data, test_data = get_split(DATASET_DIR)
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    # write to txt file
    with open("train.txt", "w") as f:
        for line in train_data:
            f.write(f'{os.path.join(".", "images", line)}\n')
    with open("val.txt", "w") as f:
        for line in val_data:
            f.write(f'{os.path.join(".", "images", line)}\n')
    with open("test.txt", "w") as f:
        for line in test_data:
            f.write(f'{os.path.join(".", "images", line)}\n')

if __name__ == "__main__":
    main()
