# src/schedulers.py
from abc import ABC, abstractmethod
import numpy as np

class LRScheduler(ABC):
    """Abstract base class for learning rate scheduling strategies."""
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    @abstractmethod
    def step(self, epoch):
        """Computes and returns the decayed learning rate for the active epoch loop."""
        pass

class StepDecay(LRScheduler):
    """Decays the learning rate by a fixed drop ratio at specified epoch intervals."""
    def __init__(self, initial_lr, drop_ratio=0.5, epochs_per_drop=50):
        super().__init__(initial_lr)
        self.drop_ratio = drop_ratio
        self.epochs_per_drop = epochs_per_drop

    def step(self, epoch):
        """Computes step-decayed learning rate based on elapsed epochs."""
        self.current_lr = self.initial_lr * (self.drop_ratio ** np.floor(epoch / self.epochs_per_drop))
        return self.current_lr

class ExponentialDecay(LRScheduler):
    """Decays the learning rate exponentially over each epoch."""
    def __init__(self, initial_lr, decay_rate=0.96):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self, epoch):
        """Computes exponentially decayed learning rate for the given epoch."""
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.current_lr