import torch


class LossNormedState(torch.nn.Module):
    def __init__(self, component_model):
        super(LossNormedState, self).__init__()
        self.scale_to_norm = component_model.scale_to_norm

    def forward(self, inputs, targets):
        loss_full = (inputs - targets) * self.scale_to_norm
        loss_point_wise = torch.square(
            torch.linalg.norm(loss_full, ord=2, dim=1, keepdim=True)
        )
        loss = torch.mean(loss_point_wise)
        return loss
