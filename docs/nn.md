# NN
## FixedAttention2d

*`class torchdish.nn.FixedAttention2d(channels: int, heads: int, H: int, W: int, r: int=1)`*

`torch.nn.Module`-style class implementing fixed attention ([Child et al. 2019](https://doi.org/10.48550/arXiv.1904.10509)) on a local neighbourhood of radius `r`. 

Arguments: 
- `channels` (`int`): the number of channels in the state
- `heads` (`int`): number of attention heads (must divide `channels`). 
- `H` (`int`): the height of the state. 
- `W` (`int`): the width of the state. 
- `r` (`int`): the radius of the (square) local neighbourhood. For example, the default value `r=1` corresponds to a 3x3 neighbourhood. 