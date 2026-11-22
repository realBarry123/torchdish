# NN
## FixedAttention2d

*`class torchdish.nn.FixedAttention2d(channels: int, heads: int, H: int, W: int, r: int=1)`*

Subclass of `nn.Module` implementing fixed attention ([Child et al. 2019](https://doi.org/10.48550/arXiv.1904.10509)) on a local neighbourhood of radius `r`. 

Arguments: 
- `channels` (`int`): the number of channels in the state
- `heads` (`int`): number of attention heads (must divide `channels`). 
- `H` (`int`): the height of the state. 
- `W` (`int`): the width of the state. 
- `r` (`int`): the radius of the (square) local neighbourhood. For example, the default value `r=1` corresponds to a 3x3 neighbourhood. 

## SobelFilter

*`class torchdish.nn.SobelFilter()`*

Subclass of `nn.Module` implementing Sobel filtering, for use in perceive functions. 

`SobelFilter.forward(x: torch.Tensor) -> torch.Tensor`: 
- Input: tensor with shape `(B, C, H, W)`
- Output: Sobel-filtered result with shape `(B, 2 * C, H, W)` where output[:, :C] are the x components of the gradient and output[:, C:] are the y components