"""
Управление random seeds для воспроизводимости результатов.
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Устанавливает random seed для всех используемых библиотек.
    
    Args:
        seed: Значение random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Для максимальной воспроизводимости (может снизить производительность)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

