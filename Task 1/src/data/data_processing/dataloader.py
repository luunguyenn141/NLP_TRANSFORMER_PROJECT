import torch
from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Tạo DataLoader cho mô hình Transformer.

    Args:
        dataset: Instance của TranslationDataset
        batch_size (int): kích thước batch
        shuffle (bool): có xáo trộn dữ liệu hay không
        num_workers (int): số worker để load dữ liệu (Windows nên để = 0)

    Returns:
        DataLoader
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
