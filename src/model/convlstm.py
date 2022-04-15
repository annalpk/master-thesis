import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    ConvLSTM unit

    Parameters:
        input_dim: int
            Number of channels of input tensor
        hidden_dim: int
            Number of channels of the hidden state, also the number of channels of the convolution output
        kernel_size: int
            Size of the convolutional kernel
        padding: int
            adds padding to the convolution
        bias: bool
            Should the bias be added or not
        """


class ConvLSTM_Unit(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding, bias=False):
        super(ConvLSTM_Unit, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        self.Wxi = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)
        self.Whi = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)

        self.Wxf = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)
        self.Whf = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)

        self.Wxo = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)
        self.Who = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)

        self.Wxc = nn.Conv2d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)
        self.Whc = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=self.kernel_size,
                             padding=self.padding, bias=self.bias)

    def forward(self, input, hidden_state, current_state):
        i = torch.sigmoid(self.Wxi(input) + self.Whi(hidden_state))
        f = torch.sigmoid(self.Wxf(input) + self.Whf(hidden_state))
        o = torch.sigmoid(self.Wxo(input) + self.Who(hidden_state))
        c = f * current_state + i * torch.tanh(self.Wxc(input) + self.Whc(hidden_state))
        h = o * torch.tanh(c)
        return h, c

    # initializes first hidden state with all zeroes
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width),
                torch.zeros(batch_size, self.hidden_dim, height, width))


class ConvLSTM(nn.Module):

    """
    The ConvLSTM Module with seq_len units

    """

    def __init__(self, input_dim, hidden_dim, kernel_size, padding, num_layers, bias, device):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers
        self.bias = bias
        self.device = device

        layer_list = []

        for i in range(0, num_layers):
            current_input_dim = self.input_dim if i == 0 else self.hidden_dim

            layer_list.append(
                ConvLSTM_Unit(input_dim=current_input_dim, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size, padding=self.padding,
                              bias=self.bias))

        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, input, hidden_state=None):
        """Parameters:
            input: Tensor of shape (Batch, Sequence Length, Channels, Height, Width)
            hidden_state: for the first convLSTM unit it is the initialized hidden state with zeroes. For every other convLSTM unit it is the hidden state calculated by the unit before the current one"""

        if hidden_state==None:
            hidden_state = self._init_hidden(batch_size=input.size(0), image_size=(input.size(-2), input.size(-1)))# initialize hidden states
        else:
            hidden_state = hidden_state

        current_input = input

        last_layer_states = []
        outputs = []
        seq_len = current_input.size(1)

        for layer_idx in range(self.num_layers):  # loop for every layer in ConvLSTM
            h,c = hidden_state[layer_idx]
            h = h.to(self.device)
            c = c.to(self.device)
            sequence_t_list = []

            for t in range(seq_len):  # loop for every step in the sequence
                h, c = self.layer_list[layer_idx](input=current_input[:, t, :, :, :], hidden_state=h, current_state=c)
                sequence_t_list.append(h)

            layer_output = torch.stack(sequence_t_list, dim=1)
            outputs.append(layer_output)
            last_layer_states.append((h, c))

        return outputs

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.layer_list[i].init_hidden(batch_size, image_size))
        return init_states