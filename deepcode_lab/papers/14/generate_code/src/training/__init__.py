# src/training/__init__.py
"""Training package initialization.

Exports the main trainer classes for easy import:

```python
from src.training import NSFGTrainer, NSFGPPTrainer
```
"""

from .nsfg_trainer import NSFGTrainer
from .nsfgpp_trainer import NSFGPPTrainer

__all__ = ["NSFGTrainer", "NSFGPPTrainer"]
