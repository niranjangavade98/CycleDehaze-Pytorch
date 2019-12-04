import torch.nn as nn
import functools

class Generator(nn.Module):
    """
    Generator model that uses ResNet architecture between upsampling & downsampling layers.
    
    Parms:
    inp        : channels in input image; default=3
    out        : channels in output image; default=3
    res_blocks : number of res-blocks in Resnet; default=6
    """
    def __init__(self,inp=3,out=3,res_blocks=6):
        
        assert (res_blocks>0), "There should be atleast 1 ResNet block"
        
        super(Generator,self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [   
                    nn.ReflectionPad2d(3),                                              #Reflection padding applied to inp image

                    nn.Conv2d(inp, 64, kernel_size=7, padding=0, bias=True),            #7X7 conv applied; 64 filters/outputs
                    norm_layer(64),                                                     #InstanceNorm2D applied
                    nn.ReLU(True),                                                      #Relu activalion applied

                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),  #downsampling layer-1
                    norm_layer(128),                                                    #
                    nn.ReLU(True),                                                      #

                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True), #downsampling layer-2
                    norm_layer(256),                                                    #
                    nn.ReLU(True),                                                      #
                ]
        
        for i in range(res_blocks):                                                     #add multiple ResNet blocks

            model +=[
                        ResnetBlock(inp_channels=256, norm_layer=norm_layer, use_dropout=False)
                    ]


        model +=[   
                    nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),  #upsampling layer-1
                    norm_layer(128),                                                                                #
                    nn.ReLU(True),                                                                                  #

                    nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),   #upsampling layer-2
                    norm_layer(64),                                                                                 #
                    nn.ReLU(True),                                                                                  #

                    nn.ReflectionPad2d(3),                                              #Reflection padding applied

                    nn.Conv2d(64, 3, kernel_size=7, padding=0),                         #7X7 conv applied; 3 filters/outputs

                    nn.Tanh()                                                           #Tanh activation function used finally

                ]
            
        self.model = nn.Sequential(*model)

    def forward(self, inp):
        """Standard forward pass"""
        return self.model(inp)
                    

class ResnetBlock(nn.Module):
    """Define a Resnet block
    
    Params:
    inp_channel      : mo. channels given as input; default=3
    norm_layer       : normalisation layer to be used
    use_dropout      : whether to use dropout or not; default=False
    """
    def __init__(self, inp_channels=256, norm_layer, use_dropout=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block ,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf

        Parameters:
            inp_channels (int)  -- the number of channels in the conv layer.
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers; default=False
        """
        super(ResnetBlock, self).__init__()
        
        res_block =[                                                                    # 1 full Resnet Block
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(inp_channels, inp_channels, kernel_size=3, padding=0, bias=True),
                        norm_layer(inp_channels),
                        nn.ReLU(True),
                        #nn.Dropout(0.5),                                               #dont use dropout- Niranjan
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(inp_channels, inp_channels, kernel_size=3, padding=0, bias=True),
                        norm_layer(inp_channels)
                    ]
        self.res_block = nn.Sequential(*res_block)

    def forward(self, inp):
        """Forward pass of thos ResNet block only (with skip connections)"""
        out = inp + self.res_block(inp)                                                 # add skip connections
        return out
