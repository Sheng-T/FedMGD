
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset_to_iid(opt, aug_data, dataType = 'train',batch_size = 256):
    data_loader = CustomDatasetIIDDataLoader(opt, aug_data, dataType, batch_size)
    dataset = data_loader.load_data()
    return dataset



def create_dataset(opt,dataType = 'train',batch_size = 256):
    data_loader = CustomDatasetDataLoader(opt,dataType,batch_size)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetIIDDataLoader():
    def __init__(self, opt,aug_data,dataType,batch_size):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_fuse_mode)
        self.dataset = dataset_class(opt, aug_data,dataType)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

class CustomDatasetDataLoader():

    def __init__(self, opt,dataType,batch_size):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt,dataType)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
