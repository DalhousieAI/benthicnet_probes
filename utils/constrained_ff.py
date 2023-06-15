# Modified from Coherent Hierarchical Multi-Label Classification Networks (https://github.com/EGiunchiglia/C-HMCNN)
# Under GPL-3.0 License

import torch
import torch.nn as nn


def get_constr_out(x, R):
    """Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R"""
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x), R.shape[1], R.shape[1])
    R_batch = R.expand(len(x), R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch * c_out.double(), dim=2)
    return final_out


class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout, R, non_lin="relu"):
        super(ConstrainedFFNNModel, self).__init__()
        self.R = R
        fc = []

        if len(hidden_dim) == 0:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, output_dim))
        else:
            fc.append(nn.BatchNorm1d(input_dim, affine=False))
            fc.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(len(hidden_dim)):
                if i == len(hidden_dim) - 1:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], output_dim))
                else:
                    fc.append(nn.BatchNorm1d(hidden_dim[i], affine=False))
                    fc.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))

        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(dropout)

        if non_lin == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x):
        module_len = len(self.fc)
        for i in range(module_len):
            if i == module_len - 1:
                x = self.fc[i](x)
            else:
                if i % 2 == 0:
                    x = self.f(self.fc[i](x))
                else:
                    x = self.f(self.fc[i](x))
                    x = self.drop(x)
        return x
