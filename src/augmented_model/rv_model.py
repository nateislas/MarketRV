import torch
import torch.nn as nn
import torch.nn.functional as F

def multitask_loss(outputs, targets, lambda_market):
    loss_stock = torch.mean((outputs[:, 0] - targets[:, 0]) ** 2)
    loss_market = torch.mean((outputs[:, 1] - targets[:, 1]) ** 2)
    return loss_stock + lambda_market * loss_market

class RVModel(nn.Module):
    def __init__(self, input_size=54, hidden_size=64, output_size=2, num_layers=1,
                 num_heads=2, dropout=0.2, device='cpu', num_symbols=100, symbol_embed_dim=16,
                 num_sectors=10, sector_embed_dim=8, num_industries=20, industry_embed_dim=8,
                 descriptive_input_size=3, descriptive_embed_dim=8, lambda_market_init=0.1):
        super(RVModel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Static embeddings
        self.symbol_embedding = nn.Embedding(num_embeddings=num_symbols, embedding_dim=symbol_embed_dim)
        self.sector_embedding = nn.Embedding(num_embeddings=num_sectors, embedding_dim=sector_embed_dim)
        self.industry_embedding = nn.Embedding(num_embeddings=num_industries, embedding_dim=industry_embed_dim)
        self.descriptive_encoder = nn.Sequential(
            nn.Linear(descriptive_input_size, descriptive_embed_dim),
            nn.ReLU()
        )

        encoder_input_dim = hidden_size * 2 + symbol_embed_dim + sector_embed_dim + industry_embed_dim + descriptive_embed_dim
        self.symbol_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, symbol_embed_dim),
            nn.ReLU()
        )

        half_input = input_size // 2

        self.gru_stock = nn.GRU(input_size=half_input, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True,
                                dropout=(dropout if num_layers > 1 else 0.0))
        self.gru_market = nn.GRU(input_size=half_input, hidden_size=hidden_size,
                                 num_layers=num_layers, batch_first=True,
                                 dropout=(dropout if num_layers > 1 else 0.0))

        self.norm_stock = nn.LayerNorm(hidden_size)
        self.norm_market = nn.LayerNorm(hidden_size)

        self.attn_stock = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_market = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads,
                                                 dropout=dropout, batch_first=True)

        self.head_combined = nn.Sequential(
            nn.Linear(hidden_size * 2 + symbol_embed_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

        # Learnable lambda_market (positive constraint via softplus)
        self._lambda_market_unconstrained = nn.Parameter(torch.tensor(lambda_market_init))

        self.init_weights()
        print(f"RVModel Initialized: hidden={hidden_size}, symbol_embed_dim={symbol_embed_dim}, device={device}")

    @property
    def lambda_market(self):
        return F.softplus(self._lambda_market_unconstrained)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    def forward(self, x, symbol_id, sector_id, industry_id, desc):
        split_idx = self.input_size // 2
        x_stock = x[:, :, :split_idx]
        x_market = x[:, :, split_idx:]

        out_stock, _ = self.gru_stock(x_stock)
        out_market, _ = self.gru_market(x_market)

        out_stock = self.norm_stock(out_stock)
        out_market = self.norm_market(out_market)

        attn_s2m, _ = self.attn_stock(out_stock, out_market, out_market)
        attn_m2s, _ = self.attn_market(out_market, out_stock, out_stock)

        cross_combined = torch.cat([attn_s2m, attn_m2s], dim=-1)
        pooled = cross_combined.mean(dim=1)

        base_symbol_vec = self.symbol_embedding(symbol_id)
        sector_vec = self.sector_embedding(sector_id)
        industry_vec = self.industry_embedding(industry_id)
        desc_vec = self.descriptive_encoder(desc)

        context_input = torch.cat([pooled, base_symbol_vec, sector_vec, industry_vec, desc_vec], dim=-1)
        symbol_vec = self.symbol_encoder(context_input)

        combined = torch.cat([pooled, symbol_vec], dim=-1)
        return self.head_combined(combined)

    def fit(self, train_loader, val_loader=None, epochs=100, lr=0.001, fold=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_model_path = f"models/best_model_fold_{fold}.pt"
        best_model = None
        print(f"Starting Training - Fold {fold+1} for {epochs} epochs")
        for epoch in range(epochs):
            self.train()
            train_losses = []
            for xb, yb, symbol_id, sector_id, industry_id, desc in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                symbol_id = symbol_id.to(self.device)
                sector_id = sector_id.to(self.device)
                industry_id = industry_id.to(self.device)
                desc = desc.to(self.device)
                optimizer.zero_grad()
                outputs = self(xb, symbol_id, sector_id, industry_id, desc)
                loss = multitask_loss(outputs, yb, self.lambda_market)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = sum(train_losses) / len(train_losses)

            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for xb_val, yb_val, symbol_id_val, sector_id_val, industry_id_val, desc_val in val_loader:
                        xb_val, yb_val = xb_val.to(self.device), yb_val.to(self.device)
                        symbol_id_val = symbol_id_val.to(self.device)
                        sector_id_val = sector_id_val.to(self.device)
                        industry_id_val = industry_id_val.to(self.device)
                        desc_val = desc_val.to(self.device)
                        val_outputs = self(xb_val, symbol_id_val, sector_id_val, industry_id_val, desc_val)
                        val_loss = multitask_loss(val_outputs, yb_val, self.lambda_market)
                        val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                if (epoch + 1) % 10 == 0:
                    print(f"Fold {fold+1}, Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | λ_market: {self.lambda_market.item():.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.state_dict(), best_model_path)
                    best_model = self.state_dict()
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | λ_market: {self.lambda_market.item():.4f}")
        if best_model is not None:
            self.load_state_dict(best_model)
            print(f"Best model for fold {fold+1} saved to {best_model_path}")
        return self

    def evaluate_model(self, data_loader, criterion=None):
        self.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb, symbol_id, sector_id, industry_id, desc in data_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                symbol_id = symbol_id.to(self.device)
                sector_id = sector_id.to(self.device)
                industry_id = industry_id.to(self.device)
                desc = desc.to(self.device)
                outputs = self(xb, symbol_id, sector_id, industry_id, desc)
                if criterion is not None:
                    loss = criterion(outputs, yb)
                    total_loss += loss.item() * xb.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(yb.cpu())
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else None
        return avg_loss, all_preds, all_targets

    def predict(self, x, symbol_id, sector_id, industry_id, desc):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            symbol_id = symbol_id.to(self.device)
            sector_id = sector_id.to(self.device)
            industry_id = industry_id.to(self.device)
            desc = desc.to(self.device)
            outputs = self(x, symbol_id, sector_id, industry_id, desc)
        return outputs.cpu()