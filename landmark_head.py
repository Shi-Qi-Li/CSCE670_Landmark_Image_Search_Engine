import torch

class LandmarkHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, multihead_num=4, layers_num=3, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.multihead_num = multihead_num
        self.transformer_layers_num = layers_num
        self.Q_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        # self.K_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        # self.V_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.condition_projection = torch.nn.Linear(input_dim, hidden_dim).to(device)
        self.K_projection_cross_attn = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
        self.V_projection_cross_attn = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
        
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=multihead_num,
            batch_first=True
        ).to(device)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=multihead_num,
            batch_first=True
        ).to(device)
        norm_layer = torch.nn.LayerNorm(hidden_dim).to(device)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers_num, norm=norm_layer).to(device)
        # mlp_layers = []
        # for i in range(layers_num):
        #     mlp_layers.append(torch.nn.Linear(hidden_dim, hidden_dim).to(device))
        #     mlp_layers.append(torch.nn.ReLU().to(device))
        # self.mlp = torch.nn.Sequential(*mlp_layers).to(device)
        
    def forward(self, cls_tokens, patch_features):
        condition_embed = self.condition_projection(cls_tokens).unsqueeze(1)  # [batch, 1, hidden_dim]
        Q = self.Q_projection(patch_features)  # [batch, num_patches, hidden_dim]
        
        transformer_output = self.transformer_encoder(Q, src_key_padding_mask=None)
        
        K_cross = self.K_projection_cross_attn(transformer_output)
        V_cross = self.V_projection_cross_attn(transformer_output)
        cross_attn_output, _ = self.cross_attention(
            query=condition_embed,
            key=K_cross,
            value=V_cross
        )
        
        # transformer_output = self.transformer_encoder(cross_attn_output)
        # print(transformer_output.shape)
        # output = self.mlp(cross_attn_output).squeeze(1)
        
        return cross_attn_output.squeeze(1)  # [batch, hidden_dim]