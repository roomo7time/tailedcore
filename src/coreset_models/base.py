import abc
import torch
from torch.utils.data import DataLoader

class BaseCore(abc.ABC):

    @abc.abstractmethod
    def fit(self, trainloader: DataLoader):
        pass

    @abc.abstractmethod
    def predict(self, images: torch.Tensor):
        pass