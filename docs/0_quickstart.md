# Quickstart

I'm running a little low on time to write a full tutorial, but I find examples to be very helpful learning tools. As such, I have provided an example implementation for a `Dish` subclass. I invite you to check out [tests/mnist_classify.py](tests/mnist_classify.py) for a more complete setup. 

```python
class MNISTClassifier(td.Dish):

    def __init__(self):
        super().__init__(
            mutable_shape=(1, CHANNELS, 28, 28),
            immutable_shape=(1, 3, 28, 28),
            train_immutable=True
        )
        self.conv1 = nn.Conv2d(
            in_channels=CHANNELS + self.immutable.shape[1], 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=CHANNELS, 
            out_channels=CHANNELS, 
            kernel_size=(3, 3), 
            padding="same"
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.steps = 32
        self.noise = 4

    def write(self, x):
        x = repeat(x, "b h w -> b c h w", c=CHANNELS)
        self.mutable = x

    def read(self):
        return self.mutable[:, :, 14, :10].mean(dim=1)

    def update(self, x):
        imm = self.immutable.expand(x.shape[0], -1, -1, -1) # expand immutable to fit batch
        r = torch.cat((x, imm), dim=1)
        r = self.conv1(r)
        r = self.relu(r)
        r = self.conv2(r)
        return x + r * 0.1
```