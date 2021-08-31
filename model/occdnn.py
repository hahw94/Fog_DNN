from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F

class Deep_Neural_Network_Target_Existence(nn.Module):
    def __init__(self, num_unit_1 = 200, num_unit_2 = 70, num_unit_3 = 30, input_num=-1, drop_out=0):
        super(Deep_Neural_Network_Target_Existence, self).__init__()
        self.fc1 = nn.Linear(input_num, num_unit_1)
        self.fc2 = nn.Linear(num_unit_1, num_unit_2)
        self.fc3 = nn.Linear(num_unit_2, num_unit_3)
        self.output = nn.Linear(num_unit_3, 1)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, **kwargs):
        fc1 = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.2))
        fc2 = self.dropout(F.leaky_relu(self.fc2(fc1), negative_slope=0.2))
        fc3 = self.dropout(F.leaky_relu(self.fc3(fc2), negative_slope=0.2))
        fc4 = sigmoid(self.output(fc3))
        return fc4