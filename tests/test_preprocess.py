from src.preprocess import CatsDogsDataset

def test_dataset_length():
    ds = CatsDogsDataset("data/train")
    assert len(ds) > 0

def test_label_assignment():
    ds = CatsDogsDataset("data/train")
    img, label = ds[0]
    assert label in [0,1]
