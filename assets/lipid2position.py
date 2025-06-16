import torch
from torch import nn

# ---------------------- Model Class ----------------------
class Lipid2Position(nn.Module):
    def __init__(self, input_size):
        super(Lipid2Position, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.05),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# ---------------------- Loss Options ----------------------
def compute_loss(pred, true, loss_type, k=1, a=None, b=None):
    
    # print(f"pred: {pred.shape}, true: {true.shape}")
    # print(f"loss_type: {loss_type}, k: {k}, a: {a}, b: {b}")
    
    mse_loss = torch.nn.functional.mse_loss(pred, true, reduction='none')
    
    if loss_type == 'weightedmse':
        assert a is not None and b is not None and k > 0
        weight = ((true < a) | (true > b))
        mse_loss *= (1 + k * weight.to(mse_loss.device))
    
    elif loss_type == 'xfocusmse':
        assert k > 0
        weight = torch.tensor([1, 0, 0] * (true.shape[0])).reshape(true.shape)
        mse_loss *= (1 + k * weight.to(mse_loss.device))
    elif loss_type == 'yfocusmse':
        assert k > 0
        weight = torch.tensor([0, 1, 0] * (true.shape[0])).reshape(true.shape)
        mse_loss *= (1 + k * weight.to(mse_loss.device))
    elif loss_type == 'zfocusmse':
        assert k > 0
        weight = torch.tensor([0, 0, 1] * (true.shape[0])).reshape(true.shape)
        mse_loss *= (1 + k * weight.to(mse_loss.device))
    else:
        assert loss_type == 'standardmse'

    return mse_loss.mean(dim=-1).mean()

