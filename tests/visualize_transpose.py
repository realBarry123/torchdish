import torch
import torchdish as td
from matrix_transpose import Transposer
model = Transposer()
model.load_state_dict(torch.load("tests/models/transposer.pt", weights_only=False))

import matplotlib.pyplot as plt
anim = td.plot_mutable(model, torch.rand(1, 8, 8))
plt.show()
td.plot_immutable(model)
plt.show()