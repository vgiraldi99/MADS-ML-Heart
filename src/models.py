from dataclasses import dataclass

import torch
from torch import Tensor, nn

@dataclass
class BaseModelSettings():
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.2
    batch_first: bool = True

@dataclass
class CNNSettings():
    #Settings for conv layers
    matrix_shape: tuple #  Shape of the insert matrix
    in_channels: int
    hidden_size: int 
    num_layers: int #  Amount of convolutional layers to add
    num_classes: int #  Amount of end classes to be determined
    pool_size: int = 2 #  Size of the Max or Average pool kernel (standard = 2, meaning halving the size)
    kernel_size: int = 3 #  Size of the Convolution kernel
    stride: int = 1 #  Amount of 'pixels' to move each frame
    padding: int = 1 #  Amount of 0's to pad around the edges
    attention: bool = False
    dense_activation: str = 'relu' #  Activation function to use, default is ReLU


class ConvBlock(nn.Module):
    """
    Standard convolution block
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride, 
                padding = padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding
            ),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1))
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Attention consists of multiple layers
        Matrix multiplication (Query, Key) -> Scale -> Mask -> SoftMax -> MatrixMultiplication(QK, V)
        """
        batch, channel, height, width = x.shape

        key = self.key(x).view(batch, -1, height*width)
        query = self.query(x).view(batch, -1, height*width).permute(0, 2, 1)

        attention = self.softmax(
            torch.bmm(query, key)
        )
        
        value = self.value(x).view(batch, -1, height * width)

        out = torch.bmm(
            value, attention.permute(0,2,1)).view(batch, channel, height, width)
        
        return out
    
class MultiHeadSelfAtt(nn.Module):
    def __init__(self, in_ch, num_heads=8): # typically as 8 basing it off Att is All You Need Paper
        super(MultiHeadSelfAtt, self).__init__()

        # should be able to divide input into different heads
        if in_ch % num_heads != 0: raise ValueError('Cannot equally divide input to different heads.')

        self.num_heads = num_heads
        self.channels_per_head = in_ch // num_heads

        self.query = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.key = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.val = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        self.linear = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.weights = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, intermediate=False):
        N, ch, h, w = x.shape

        query = self.query(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w
        key = self.key(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w
        val = self.val(x).view(N, self.num_heads, self.channels_per_head, h*w) # N, N_h, C_h, h*w

        att = self.softmax(
            (key.mT@query)*self.channels_per_head**-0.5, # N, N_h, h*w, h*w
        )

        out = att@val.mT #N, N_h, h*w, h*w x N, N_h, C_h, h*w --> N, N_h, h*w, C_h
        out = out.mT # N, N_h, C_h, h*w
        # https://stackoverflow.com/questions/66750391/runtimeerror-view-size-is-not-compatible-with-input-tensors-size-and-stride-a
        out = out.contiguous().view(N, ch, h, w)

        out = self.linear(out)
        out = self.weights*out + x

        if intermediate: return out, att, query, key
        else: return out

class ConvBlocks(nn.Module):
    def __init__(self, config: CNNSettings) -> None:
        super().__init__()
        self.config = config
        
        #  First convolution to get input channnels to hidden size
        self.convolutions = nn.ModuleList([
            ConvBlock(in_channels = config.in_channels, 
                      out_channels = config.hidden_size,
                      kernel_size = config.kernel_size,
                      stride = config.stride,
                      padding = config.padding)
        ])

        # Add convolution blocks based on given amount of layers
        num_maxpools = 0
        for i in range(config.num_layers):
            self.convolutions.extend([ 
                #Add new convolution layer
                ConvBlock(
                    in_channels = config.hidden_size, # Add hidden size instead of in_channel
                    out_channels = config.hidden_size,
                    kernel_size = config.kernel_size,
                    stride = config.stride,
                    padding = config.padding)
            ])

            if i % 2 == 0:
                num_maxpools += 1
                self.convolutions.append(nn.MaxPool2d(config.pool_size, config.pool_size))
                nn.MultiheadAttention
                if config.attention: self.convolutions.append(MultiHeadSelfAtt(config.hidden_size))
        
        # Calculates the endsize by dividing the shape size by the amount of poolsize decrease per maxpool
        matrix_size = \
        (config.matrix_shape[0] // (config.pool_size**num_maxpools)) * \
        (config.matrix_shape[1] // (config.pool_size**num_maxpools))

        print(f"Calculated matrix size: {matrix_size}")
        print(f"Calculated flatten size: {matrix_size * config.hidden_size}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(matrix_size * config.hidden_size, config.hidden_size),
            self._get_activation(config.dense_activation),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x
    
    def _get_activation(self, activation_function: str):
        activation = activation_function.lower()

        activation_dict = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }

        if activation not in activation_dict.keys():
            raise ValueError(f"Unsupported activation function: {activation}")
        
        return activation_dict[activation]


#----------------------------------------------------------------
#############################
# RNN Nation                #
#############################

@dataclass
class GRUSettings(BaseModelSettings):
    attention: bool = True
    attention_dropout: float = 0.2
    attention_heads: int = 4

class GRUmodel(nn.Module):
    def __init__(self, config: GRUSettings) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size = config.input_size,
            hidden_size = config.hidden_size,
            dropout = config.dropout,
            batch_first = config.batch_first,
            num_layers = config.num_layers
        )
        self.attention = nn.MultiheadAttention(
            embed_dim = config.hidden_size,
            num_heads = config.attention_heads,
            dropout = config.attention_dropout,
            batch_first = config.batch_first,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        if self.config.attention: x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
    
@dataclass
class LSTMSettings(BaseModelSettings):
    attention_dropout: float = 0.2
    attention: bool = True
    attention_heads: int = 4

class LSTMmodel(nn.Module):
    def __init__(self, config: LSTMSettings) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.LSTM(
            input_size = config.input_size,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            batch_first = config.batch_first,
            dropout = config.dropout
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)
        self.attention = nn.MultiheadAttention(
            embed_dim = config.hidden_size,
            num_heads = config.attention_heads,
            dropout = config.attention_dropout,
            batch_first = config.batch_first,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        if self.config.attention: 
            x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
