import torch.nn as nn


class MultiLabelFFNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MultiLabelFFNNModel, self).__init__()

        self.num_classes = output_dim
        fc = []
        dims = [input_dim] + hidden_dim + [self.num_classes]

        for i in range(len(dims) - 1):
            fc.append(nn.BatchNorm1d(dims[i], affine=False))
            fc.append(nn.Linear(dims[i], dims[i + 1]))

        self.fc = nn.ModuleList(fc)
        self.drop = nn.Dropout(dropout)
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
