# Notebooks

Vous pouvez expérimenter ici (Jupyter). Le pipeline d’entraînement principal est dans `scripts/train_fight.py` et `main.py train`.

Exemple de cellule :

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))
from src.behavior.fight_cnn_lstm import FightCNNLSTM
import torch
m = FightCNNLSTM()
print(m)
```
