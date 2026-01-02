from datasets import load_dataset

# Return the original LongLaMP dataset
def load_orgin_data(dataset_name: str = "abstract_generation_user"):
    ds = load_dataset("LongLaMP/LongLaMP", dataset_name)
    if dataset_name == "abstract_generation_user":
        # For abstract_generation_user, there is no validation set
        return ds["train"], ds["test"], ds["val"]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

if __name__ == "__main__":
    train_ds, test_ds, val_ds = load_orgin_data()
    print(val_ds[0])