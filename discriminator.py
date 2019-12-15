import torch.nn as nn
import functools

class Discriminator(nn.Module):
    """
    Discriminator class
    """
    def __init__(self,inp=3,out=1):
        """
        Initializes the PatchGAN model with 3 layers as discriminator

        Args:
        inp: number of input image channels
        out: number of output image channels
        """

        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model =    [
                        nn.Conv2d(inp, 64, kernel_size=4, stride=2, padding=1),             #input 3 channels
                        nn.LeakyReLU(0.2,True),

                        nn.Conv2d(64, 128, kernal_size=4, stride=2, padding=1, bias=True),
                        norm_layer(128),
                        nn.LeakyReLU(0.2, True),

                        nn.Conv2d(128, 256, kernal_size=4, stride=2, padding=1, bias=True),
                        norm_layer(256),
                        nn.LeakyReLU(0.2, True),

                        nn.Conv2d(256, 512, kernal_size=4, stride=1, padding=1, bias=True),
                        norm_layer(512),
                        nn.LeakyReLU(0.2, True),

                        nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)             #output only 1 channel (prediction map)
                    ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            Feed forward the image produced by generator through discriminator

            Args:
            input: input image

            Returns:
            outputs prediction map with 1 channel 
        """
        return self.model(input)
