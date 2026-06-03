# models/schedulers.py
from abc import ABC, abstractmethod
import numpy as np

class LRScheduler(ABC):
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    @abstractmethod
    def step(self, epoch):
        """Computes and returns the decayed learning rate for the active epoch loop."""
        pass

class StepDecay(LRScheduler):
    def __init__(self, initial_lr, drop_ratio=0.5, epochs_per_drop=50):
        super().__init__(initial_lr)
        self.drop_ratio = drop_ratio
        self.epochs_per_drop = epochs_per_drop

    def step(self, epoch):
        self.current_lr = self.initial_lr * (self.drop_ratio ** np.floor(epoch / self.epochs_per_drop))
        return self.current_lr

class ExponentialDecay(LRScheduler):
    def __init__(self, initial_lr, decay_rate=0.96):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self, epoch):
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.current_lr