import inspect
import torch
from torch.utils.data import Dataset

from whos_that_pokemon.models import UNet
from whos_that_pokemon import pipeline


class TinyFakeDataset(Dataset):
    def __init__(self, n=4):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.randn(3, 64, 64), torch.randint(0, 2, (1, 64, 64)).float()


def test_bug_1():
    torch.manual_seed(0)
    model = UNet(base=8)
    model.eval()

    x1 = torch.randn(1, 3, 64, 64)
    x2 = x1.clone()
    x2[:, :, :8, :8] += 5.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    diff = (out1 - out2).abs().mean().item()
    assert diff > 1e-4, "test failed"


def test_bug_2():
    from whos_that_pokemon.train_segmentation import train_segmentation_model
    import whos_that_pokemon.config as config
    import torch.optim as optim

    call_count = {"n": 0}
    real_adam = optim.Adam

    def counting_adam(*args, **kwargs):
        call_count["n"] += 1
        return real_adam(*args, **kwargs)

    train_ds = TinyFakeDataset(4)
    val_ds = TinyFakeDataset(2)

    original_epochs, original_batch = config.EPOCHS, config.BATCH_SIZE
    config.EPOCHS, config.BATCH_SIZE = 3, 2

    original_adam = optim.Adam
    optim.Adam = counting_adam

    try:
        train_segmentation_model(train_ds, val_ds, device=torch.device("cpu"))
    finally:
        config.EPOCHS, config.BATCH_SIZE = original_epochs, original_batch
        optim.Adam = original_adam

    assert call_count["n"] == 1, "test failed"


def _bug_3_check():
    source = inspect.getsource(pipeline.load_pipeline)
    return "sorted(" in source


def test_bug_3():
    assert _bug_3_check(), "test failed"