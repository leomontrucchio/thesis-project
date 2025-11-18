import torch
from torchsummary import summary

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU, reduction_factor = 1):  # GELU for ViT student and SiLU for LLM student
        super().__init__()

        self.act_fcn = act_layer()

        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x
    


class ResidualFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor = 1):
        super().__init__()
        self.act_fcn = act_layer()

        hidden_features = int((in_features + out_features) // (2 * reduction_factor))

        self.residual_block = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.LayerNorm(hidden_features),
            act_layer(),
            torch.nn.Linear(hidden_features, out_features),
            #act_layer(),
        )

        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = self.residual_block(x)

        shortcut_x = self.shortcut(x)

        return shortcut_x + residual




class NormalizedFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor=1, use_norm=True):
        super().__init__()

        self.act_fcn = act_layer()
        self.use_norm = use_norm

        hidden_features = int((in_features + out_features) / (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

        if self.use_norm:
            self.norm1 = torch.nn.LayerNorm(hidden_features)
            self.norm2 = torch.nn.LayerNorm(hidden_features)

    def forward(self, x):
        x = self.input(x)
        if self.use_norm:
            x = self.norm1(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        if self.use_norm:
            x = self.norm2(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x



class FeatureProjectionBottleneckMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, reduction_factor = 1, act_layer=torch.nn.GELU):
        super().__init__()

        self.act_fcn = act_layer()

        intermediate1_dim = int((in_features + out_features) // (2 * reduction_factor))
        intermediate2_dim = int((in_features + out_features) // (3 * reduction_factor))
        bottleneck_dim = int((in_features + out_features) // (4 * reduction_factor))

        self.input = torch.nn.Linear(in_features, intermediate1_dim)
        self.encoder = torch.nn.Linear(intermediate1_dim, intermediate2_dim)
        self.bottleneck = torch.nn.Linear(intermediate2_dim, bottleneck_dim)
        self.decoder1 = torch.nn.Linear(bottleneck_dim, intermediate2_dim)
        self.decoder2 = torch.nn.Linear(intermediate2_dim, intermediate1_dim)
        self.output = torch.nn.Linear(intermediate1_dim, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.encoder(x)
        x = self.act_fcn(x)

        x = self.bottleneck(x)
        x = self.act_fcn(x)

        x = self.decoder1(x)
        x = self.act_fcn(x)

        x = self.decoder2(x)
        x = self.act_fcn(x)

        x = self.output(x)
        
        return x



class RegularizedFeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, act_layer=torch.nn.GELU, reduction_factor=1, dropout_prob=0.3):
        super().__init__()
        self.act_fcn = act_layer()

        hidden_features = int((in_features + out_features) / (2 * reduction_factor))

        self.input = torch.nn.Linear(in_features, hidden_features)
        self.projection = torch.nn.Linear(hidden_features, hidden_features)
        self.output = torch.nn.Linear(hidden_features, out_features)

        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)
        x = self.dropout(x)

        x = self.projection(x)
        x = self.act_fcn(x)
        x = self.dropout(x)

        x = self.output(x)

        return x



if __name__ == "__main__":
    IN_FEATURES = OUT_FEATURES = 1152
    REDUCTION = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = ResidualFeatureProjectionMLP(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        reduction_factor=REDUCTION
    ).to(device)

    summary(student_model, input_size=(IN_FEATURES,), batch_dim=0, verbose=1)
