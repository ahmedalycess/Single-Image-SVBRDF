import torch
import torch.nn as nn
import utils

class LayerBootstrapping:
    """
    Boot strapper for sharing setup and initialization of convolutional and linear layers.
        Args:
            use_convolution_bias:   If True, the convolutional layers will have a bias term.
            use_linear_bias:        If True, the linear layers will have a bias term.
            initialize_weights:     If True, the weights of the layers will be initialized.
            convolution_init_scale: The scale of the initialization of the convolutional layers.
            linear_init_scale:      The scale of the initialization of the linear layers
        functions:
            initialize_module: Initializes the weights of the given module.
    """

    def __init__(self, use_convolution_bias=False, use_linear_bias=False, initialize_weights=True, convolution_init_scale=0.02, linear_init_scale=0.01):
        self.use_convolution_bias   = use_convolution_bias
        self.use_linear_bias        = use_linear_bias
        self.initialize_weights     = initialize_weights
        self.convolution_init_scale = convolution_init_scale
        self.linear_init_scale      = linear_init_scale

    def initialize_module(self, m):
        if self.initialize_weights:
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, 0.0, self.linear_init_scale * torch.sqrt(torch.tensor(1.0 / float(m.in_features))))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight, 0.0, self.convolution_init_scale)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        return m

class MergeLayer(nn.Module):
    """
    Merging layer that adds the global track to the input tensor.
        Args:
            bootstrapping:   setup and initialization of fully connected layer using bootstrapping.linear
            channel_count:   The number of channels of the input tensor.\
        forward:
            Adds the global track to the input tensor:
                - passes the global track through a fully connected layer
                - adds the global track to the input tensor
    """

    def __init__(self, bootstrapping, channel_count):
        super(MergeLayer, self).__init__()
        self.channel_count   = channel_count
        self.fully_connected = nn.Linear(self.channel_count, self.channel_count, bias=bootstrapping.use_linear_bias)
        self.apply(bootstrapping.initialize_module)

    def forward(self, x, global_track):
        if global_track is not None:
            global_track = self.fully_connected(global_track)
            x = torch.add(x, global_track.unsqueeze(-1).unsqueeze(-1))
        
        return x

class InterconnectedConvLayer(nn.Module):
    """
    layer that is interconnected with the global track:
        Args:
            bootstrapping:             setup and initialization of convolutional layer using bootstrapping.conv
            conv:                      The convolutional layer
            conv_output_channels: The number of output channels of the convolutional layer
            use_instance_norm:         If True, the layer uses instance normalization
            use_activation:            If True, the layer uses activation function
        Blocks:
            - Leaky ReLU activation function
            - Convolutional layer
            - Instance normalization
            - Merging layer
        Forward:
            - Applies the Leaky ReLU activation function
            - Applies the convolutional layer
            - Applies the instance normalization
            - Merges the global track with the input tensor
    """

    def __init__(self, bootstrapping, conv, conv_output_channels, use_instance_norm, use_activation=True):
        super(InterconnectedConvLayer, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.2) if use_activation else None

        self.conv                      = conv
        self.conv_output_channels = conv_output_channels

        self.norm       = nn.InstanceNorm2d(self.conv_output_channels, 1e-5, affine=True) if use_instance_norm else None

        self.merge      = MergeLayer(bootstrapping, self.conv_output_channels)

        self.apply(bootstrapping.initialize_module)

    def forward(self, x, global_track):
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)

        x = self.conv(x)

        mean = torch.mean(x, dim=(2,3), keepdim=False)

        if self.norm is not None:
            x = self.norm(x)

        x = self.merge(x, global_track)
        
        return x, mean

class EncodingLayer(nn.Module):
    """
    Encoding layer that performs a convolution to downsample the input and is connected to the global track.
        Args:
            bootstrapping:         setup and initialization of convolutional layer using bootstrapping.conv
            input_channels:        The number of channels of the input tensor
            output_channels:  The number of output channels of the convolutional layer
            use_instance_norm:     If True, the layer uses instance normalization
            use_activation:        If True, the layer uses activation function
        Blocks:
            - Convolutional layer
            - InterconnectedConvLayer
        Forward:
            - Applies the interconnected convolutional layer
    """

    def __init__(self, bootstrapping, input_channels, output_channels, use_instance_norm, use_activation=True):
        super(EncodingLayer, self).__init__()

        self.output_channels = output_channels

        conv       = nn.Conv2d(input_channels, output_channels, (4, 4), stride=2, padding=(1,1), bias=bootstrapping.use_convolution_bias)
        self.conv  = InterconnectedConvLayer(bootstrapping, conv, output_channels, use_instance_norm, use_activation)

        self.apply(bootstrapping.initialize_module)

    def forward(self, x, global_track):
        return self.conv(x, global_track)

class DecodingLayer(nn.Module):
    """
    Decoding layer that performs a convolution to upsample the input and is connected to the global track.
        Args:
            bootstrapping:         setup and initialization of convolutional layer using bootstrapping.conv
            input_channels:        The number of channels of the input tensor
            output_channels:  The number of output channels of the convolutional layer
            use_instance_norm:     If True, the layer uses instance normalization
            use_dropout:           If True, the layer uses dropout
            use_activation:        If True, the layer uses activation function
        Blocks:
            - Upsampling
            - Convolutional layer
            - InterconnectedConvLayer
            - Dropout
        Forward:
            - Concatenates the input tensor with the skip connected tensor
            - Applies the interconnected convolutional layer
            - Applies dropout
            
    """

    def __init__(self, bootstrapping, input_channels, output_channels, use_instance_norm, use_dropout, use_activation=True):
        super(DecodingLayer, self).__init__()
        
        self.output_channels = output_channels

        deconv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(input_channels,  output_channels, (4, 4), bias=bootstrapping.use_convolution_bias),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(output_channels, output_channels, (4, 4), bias=bootstrapping.use_convolution_bias),
        )
        self.deconv  = InterconnectedConvLayer(bootstrapping, deconv, output_channels, use_instance_norm, use_activation)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

        self.apply(bootstrapping.initialize_module)

    def forward(self, x, skip_connected_tensor, global_track):
        if skip_connected_tensor is not None:
            x = torch.cat((x, skip_connected_tensor), dim=1)

        x, mean = self.deconv(x, global_track)

        if self.dropout is not None:
            x = self.dropout(x)
        
        return x, mean

class GlobalTrackLayer(nn.Module):
    """
    Layer that maintains a global track of the input data and propagates it through the network.
        Args:
            bootstrapping:     setup and initialization of fully connected layer using bootstrapping.linear
            input_channels:    The number of channels of the input tensor
            output_channels:   The number of output channels of the fully connected layer
        Forward:
            - Concatenates the global track with the local mean
            - Applies the fully connected layer
            - Applies the SELU activation function
    """

    def __init__(self, bootstrapping, input_channels, output_channels):
        super(GlobalTrackLayer, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.fully_connected = nn.Linear(input_channels, output_channels, bias=bootstrapping.use_linear_bias)
        self.selu            = nn.SELU()

        self.apply(bootstrapping.initialize_module)

    def forward(self, local_mean, global_track):
        if global_track is not None:
            global_track = torch.cat((global_track, local_mean), dim=1)
        else:
            global_track = local_mean

        return self.selu(self.fully_connected(global_track))

class Generator(nn.Module):
    """
    Generator network that generates the SVBRDF from an input image.
        Args:
            output_channels: The number of output channels of the generator
            number_of_filters: The number of filters in the generator
        Blocks:
            - EncodingLayer
            - DecodingLayer
            - GlobalTrackLayer
        Forward:
            - Encoding with passing means through the global track
            - Decoding with passing means through the global track
    """
    input_channels  = 3
    output_channels = 64 
    filters_count        = 64 # "ngf" in the original code

    def __init__(self, output_channels, number_of_filters = 64):
        super(Generator, self).__init__()
        self.number_of_filters    = number_of_filters
        self.output_channels = output_channels 

        # Use a bootstrapper for sharing setup and initialization of convolutional and linear layers in the encoder/decoder
        encdec_bootstrap = LayerBootstrapping(use_convolution_bias=False, use_linear_bias=False, initialize_weights=True, convolution_init_scale=0.02, linear_init_scale=0.01)

        
        self.enc1 = EncodingLayer(encdec_bootstrap, self.input_channels,   self.number_of_filters    , False, False) # encoder_1: [batch, 256, 256, 3      ] => [batch, 128, 128, ngf    ]
        self.enc2 = EncodingLayer(encdec_bootstrap, self.enc1.output_channels, self.number_of_filters * 2,  True)        # encoder_2: [batch, 128, 128, ngf    ] => [batch,  64,  64, ngf * 2]
        self.enc3 = EncodingLayer(encdec_bootstrap, self.enc2.output_channels, self.number_of_filters * 4,  True)        # encoder_3: [batch,  64,  64, ngf * 2] => [batch,  32,  32, ngf * 4]
        self.enc4 = EncodingLayer(encdec_bootstrap, self.enc3.output_channels, self.number_of_filters * 8,  True)        # encoder_4: [batch,  32,  32, ngf * 4] => [batch,  16,  16, ngf * 8]
        self.enc5 = EncodingLayer(encdec_bootstrap, self.enc4.output_channels, self.number_of_filters * 8,  True)        # encoder_5: [batch,  16,  16, ngf * 8] => [batch,   8,   8, ngf * 8]
        self.enc6 = EncodingLayer(encdec_bootstrap, self.enc5.output_channels, self.number_of_filters * 8,  True)        # encoder_6: [batch,   8,   8, ngf * 8] => [batch,   4,   4, ngf * 8]
        self.enc7 = EncodingLayer(encdec_bootstrap, self.enc6.output_channels, self.number_of_filters * 8,  True)        # encoder_7: [batch,   4,   4, ngf * 8] => [batch,   2,   2, ngf * 8]
        self.enc8 = EncodingLayer(encdec_bootstrap, self.enc7.output_channels, self.number_of_filters * 8, False)        # encoder_8: [batch,   2,   2, ngf * 8] => [batch,   1,   1, ngf * 8]                  

        self.dec8 = DecodingLayer(encdec_bootstrap, self.number_of_filters * 8,         self.number_of_filters * 8,  True,  True) # decoder_8: [batch,  1,  1,       ngf * 8] => [batch,   2,   2, ngf * 8]
        self.dec7 = DecodingLayer(encdec_bootstrap, 2 * self.dec8.output_channels, self.number_of_filters * 8,  True,  True) # decoder_7: [batch,  2,  2,   2 * ngf * 8 ] => [batch,   4,   4, ngf * 8]
        self.dec6 = DecodingLayer(encdec_bootstrap, 2 * self.dec7.output_channels, self.number_of_filters * 8,  True,  True) # decoder_6: [batch,  4,  4,   2 * ngf * 8 ] => [batch,   8,   8, ngf * 8] 
        self.dec5 = DecodingLayer(encdec_bootstrap, 2 * self.dec6.output_channels, self.number_of_filters * 8,  True, False) # decoder_5: [batch,  8,  8,   2 * ngf * 8 ] => [batch,  16,  16, ngf * 8]
        self.dec4 = DecodingLayer(encdec_bootstrap, 2 * self.dec5.output_channels, self.number_of_filters * 4,  True, False) # decoder_4: [batch, 16, 16,   2 * ngf * 8 ] => [batch,  32,  32, ngf * 4]
        self.dec3 = DecodingLayer(encdec_bootstrap, 2 * self.dec4.output_channels, self.number_of_filters * 2,  True, False) # decoder_3: [batch, 32, 32,   2 * ngf * 4 ] => [batch,  64,  64, ngf * 2]
        self.dec2 = DecodingLayer(encdec_bootstrap, 2 * self.dec3.output_channels, self.number_of_filters    ,  True, False) # decoder_2: [batch, 64, 64,   2 * ngf * 2 ] => [batch, 128, 128, ngf    ]
        self.dec1 = DecodingLayer(encdec_bootstrap, 2 * self.dec2.output_channels, self.output_channels , False, False) # decoder_1: [batch, 128, 128, 2 * ngf     ] => [batch, 256, 256, 64     ]                   

        # Use a bootstrapper for sharing setup and initialization of convolutional and linear layers in the global track
        gt_boostrap = LayerBootstrapping(use_linear_bias=True, initialize_weights=True, linear_init_scale=1.0)

        self.gte1 = GlobalTrackLayer(gt_boostrap, self.input_channels,       self.enc2.output_channels) 
        self.gte2 = GlobalTrackLayer(gt_boostrap, 2 * self.enc2.output_channels, self.enc3.output_channels) 
        self.gte3 = GlobalTrackLayer(gt_boostrap, 2 * self.enc3.output_channels, self.enc4.output_channels) 
        self.gte4 = GlobalTrackLayer(gt_boostrap, 2 * self.enc4.output_channels, self.enc5.output_channels) 
        self.gte5 = GlobalTrackLayer(gt_boostrap, 2 * self.enc5.output_channels, self.enc6.output_channels)
        self.gte6 = GlobalTrackLayer(gt_boostrap, 2 * self.enc6.output_channels, self.enc7.output_channels) 
        self.gte7 = GlobalTrackLayer(gt_boostrap, 2 * self.enc7.output_channels, self.enc8.output_channels) 
        self.gte8 = GlobalTrackLayer(gt_boostrap, 2 * self.enc8.output_channels, self.dec8.output_channels) 

        self.gtd8 = GlobalTrackLayer(gt_boostrap, 2 * self.dec8.output_channels, self.dec7.output_channels) 
        self.gtd7 = GlobalTrackLayer(gt_boostrap, 2 * self.dec7.output_channels, self.dec6.output_channels)
        self.gtd6 = GlobalTrackLayer(gt_boostrap, 2 * self.dec6.output_channels, self.dec5.output_channels)
        self.gtd5 = GlobalTrackLayer(gt_boostrap, 2 * self.dec5.output_channels, self.dec4.output_channels) 
        self.gtd4 = GlobalTrackLayer(gt_boostrap, 2 * self.dec4.output_channels, self.dec3.output_channels)
        self.gtd3 = GlobalTrackLayer(gt_boostrap, 2 * self.dec3.output_channels, self.dec2.output_channels) 
        self.gtd2 = GlobalTrackLayer(gt_boostrap, 2 * self.dec2.output_channels, self.dec1.output_channels) 
        self.gtd1 = GlobalTrackLayer(gt_boostrap, 2 * self.dec1.output_channels, self.output_channels)

    def forward(self, input):

        input_mean = torch.mean(input, dim=(2,3), keepdim=False)

        # Encoding
        down1, _          = self.enc1(input,      None)
        global_track      = self.gte1(input_mean, None)
        down2, down2_mean = self.enc2(down1,      global_track)
        global_track      = self.gte2(down2_mean, global_track)
        down3, down3_mean = self.enc3(down2,      global_track)
        global_track      = self.gte3(down3_mean, global_track)
        down4, down4_mean = self.enc4(down3,      global_track)
        global_track      = self.gte4(down4_mean, global_track)
        down5, down5_mean = self.enc5(down4,      global_track)
        global_track      = self.gte5(down5_mean, global_track)
        down6, down6_mean = self.enc6(down5,      global_track)
        global_track      = self.gte6(down6_mean, global_track)
        down7, down7_mean = self.enc7(down6,      global_track)
        global_track      = self.gte7(down7_mean, global_track)
        down8, down8_mean = self.enc8(down7,      global_track)
        global_track      = self.gte8(down8_mean, global_track)

        # Decoding
        up8, up8_mean = self.dec8(down8, None, global_track)
        global_track  = self.gtd8(up8_mean,    global_track)
        up7, up7_mean = self.dec7(up8, down7,  global_track)
        global_track  = self.gtd7(up7_mean,    global_track)
        up6, up6_mean = self.dec6(up7, down6,  global_track)
        global_track  = self.gtd6(up6_mean,    global_track)
        up5, up5_mean = self.dec5(up6, down5,  global_track)
        global_track  = self.gtd5(up5_mean,    global_track)
        up4, up4_mean = self.dec4(up5, down4,  global_track)
        global_track  = self.gtd4(up4_mean,    global_track)
        up3, up3_mean = self.dec3(up4, down3,  global_track)
        global_track  = self.gtd3(up3_mean,    global_track)
        up2, up2_mean = self.dec2(up3, down2,  global_track)
        global_track  = self.gtd2(up2_mean,    global_track)
        up1, up1_mean = self.dec1(up2, down1,  global_track)
        global_track  = self.gtd1(up1_mean,    global_track)

        return up1, global_track

class SVBRDFNetwork(nn.Module):
    """
    SVBRDF network that generates the SVBRDF from an input image.
        Blocks:
            - Generator
            - Activation function
        Forward:
            - Generates the SVBRDF from the input image
            - Applies the activation function
            - Decodes the SVBRDF into the 12 channels
            - De-processes the diffuse, roughness, and specular albedo
    """
    def __init__(self):
        super(SVBRDFNetwork, self).__init__()

        self.generator  = Generator(9)
        self.activation = nn.Tanh()

    def forward(self, input):

        svbrdf, _ = self.generator(input)
        svbrdf    = self.activation(svbrdf)

        # 9 channel SVBRDF to 12 channels
        svbrdf    = utils.decode_svbrdf(svbrdf) 

        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
        diffuse   = utils.deprocess(diffuse)
        roughness = utils.deprocess(roughness)
        specular  = utils.deprocess(specular)

        return utils.pack_svbrdf(normals, diffuse, roughness, specular)
