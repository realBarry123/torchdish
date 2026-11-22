import torch
import torchdish as td
from matrix_transpose import Transposer
from matplotlib.patches import Rectangle

model = Transposer()
model.load_state_dict(torch.load("tests/models/mover_16s_16c+3.pt", weights_only=False))

import matplotlib.pyplot as plt
anim = td.plot_mutable(model, (torch.rand(1, 8, 8) > 0.5).float(), normalize_mode="clip")
axis = anim._fig.axes[0]
axis.add_patch(
    Rectangle(
        (15.5, 7.5),
        8,
        8,
        fill=False,
        edgecolor="yellow",
        linewidth=1.5,
    )
)
plt.show()
td.plot_immutable(model)
plt.show()