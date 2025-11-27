import torch

class LandmarkHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, multihead_num=4, layers_num=3, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.multihead_num = multihead_num
        self.transformer_layers_num = layers_num
        self.Q_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.K_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.V_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        ).to(device)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=multihead_num,
            batch_first=True
        ).to(device)
        norm_layer = torch.nn.LayerNorm(hidden_dim).to(device)
        # self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers_num, norm=norm_layer).to(device)
        mlp_layers = []
        for i in range(layers_num):
            mlp_layers.append(torch.nn.Linear(hidden_dim, hidden_dim).to(device))
            mlp_layers.append(torch.nn.ReLU().to(device))
        self.mlp = torch.nn.Sequential(*mlp_layers).to(device)
        
    def forward(self, cls_tokens, patch_features):
        Q = self.Q_projection(cls_tokens).unsqueeze(1)  # [batch, 1, hidden_dim]
        K = self.K_projection(patch_features)  # [batch, num_patches, hidden_dim]
        V = self.V_projection(patch_features)  # [batch, num_patches, hidden_dim]
        
        print(Q.shape, K.shape, V.shape)
        
        cross_attn_output, _ = self.cross_attention(
            query=Q,
            key=K,
            value=V
        )
        print(cross_attn_output.shape)
        
        # transformer_output = self.transformer_encoder(cross_attn_output)
        # print(transformer_output.shape)
        output = self.mlp(cross_attn_output).squeeze(1)
        
        return output