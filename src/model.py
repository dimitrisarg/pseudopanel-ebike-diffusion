"""Conditional variational autoencoder used for synthetic travel panels.

The model jointly represents mode-specific trip counts, trip lengths, and
daily trip purpose while conditioning on numeric and categorical covariates.
It supports learned embeddings or one-hot categorical inputs and Gamma, Beta,
or log-normal length distributions. Training is provided by ``train_vae``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Beta, LogNormal
import torch.optim as optim
import copy


class CVAE_StructuralZeros(nn.Module):
    def __init__(self, num_modes, purpose_dim, trip_max_num,
                 cond_num_dim=0, cond_cat_dims=None,
                 hidden_layers=[128, 64], latent_dim=[16],
                 use_cond_embedding=True, cond_embedding_dims=None,
                 max_embed_dim=50, dropout_rate=0,
                 use_purpose_embedding=True,
                 length_distribution='gamma'): 
        """
        Enhanced CVAE with multiple distribution options for trip lengths.
        
        Args:
            length_distribution: 'gamma', 'beta', or 'lognormal'
        """
        super().__init__()
        self.num_modes = num_modes
        self.purpose_dim = purpose_dim
        self.latent_dim = latent_dim
        self.cond_cat_dims = cond_cat_dims or []
        self.cond_num_dim = cond_num_dim
        self.use_purpose_embedding = use_purpose_embedding
        self.length_distribution = length_distribution

        # --- Purpose embedding ---
        if use_purpose_embedding:
            purpose_embed_dim = min(max_embed_dim, (purpose_dim + 1) // 2)
            self.purpose_embedding = nn.Embedding(purpose_dim, purpose_embed_dim)
            self.purpose_input_dim = purpose_embed_dim
        else:
            self.purpose_embedding = None
            self.purpose_input_dim = purpose_dim

        # --- Conditional embeddings ---
        if use_cond_embedding and cond_cat_dims:
            if cond_embedding_dims is None:
                cond_embedding_dims = auto_select_embedding_dims(cond_cat_dims, max_embed_dim)
            self.cond_embeddings = nn.ModuleList([
                nn.Embedding(cat_dim, emb_dim)
                for cat_dim, emb_dim in zip(cond_cat_dims, cond_embedding_dims)
            ])
            embedded_cat_dim = sum(cond_embedding_dims)
        else:
            self.cond_embeddings = None
            embedded_cat_dim = 0

        self.total_cond_dim = embedded_cat_dim + cond_num_dim

        # --- Encoder ---
        input_dim = num_modes + num_modes + self.purpose_input_dim + self.total_cond_dim
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_layers[-1], latent_dim)

        # --- Decoder ---
        input_dim = latent_dim + self.total_cond_dim
        decoder_layers = []
        prev_dim = input_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.decoder_output_dim = hidden_layers[0]

        # --- Outputs ---
        self.fc_trip_logits = nn.ModuleList([
            nn.Linear(self.decoder_output_dim, trip_max_num) for _ in range(num_modes)
        ])
        
        # Distribution parameters based on choice
        if length_distribution in ['gamma', 'beta']:
            self.fc_param1 = nn.ModuleList([
                nn.Linear(self.decoder_output_dim, 1) for _ in range(num_modes)
            ])
            self.fc_param2 = nn.ModuleList([
                nn.Linear(self.decoder_output_dim, 1) for _ in range(num_modes)
            ])
        elif length_distribution == 'lognormal':
            self.fc_mu_len = nn.ModuleList([
                nn.Linear(self.decoder_output_dim, 1) for _ in range(num_modes)
            ])
            self.fc_sigma_len = nn.ModuleList([
                nn.Linear(self.decoder_output_dim, 1) for _ in range(num_modes)
            ])
        
        self.fc_purpose = nn.Linear(self.decoder_output_dim, purpose_dim)

        # Minimum values for stability
        self.min_param = 1e-6

    def embed_conditionals(self, x_num, x_cat):
        """Embed categorical conditionals and concatenate with numeric conditionals"""
        conds = []
        
        if self.cond_embeddings is not None and x_cat is not None:
            embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.cond_embeddings)]
            conds.append(torch.cat(embedded, dim=1))
            
        if self.cond_num_dim > 0 and x_num is not None:
            conds.append(x_num)
            
        if conds:
            return torch.cat(conds, dim=1)
        else:
            batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)
            return torch.zeros(batch_size, self.total_cond_dim, 
                             device=x_num.device if x_num is not None else x_cat.device)

    def process_purpose(self, x_purpose):
        """Process purpose input (either embedding or one-hot encoding)"""
        if self.use_purpose_embedding:
            return self.purpose_embedding(x_purpose)
        else:
            return F.one_hot(x_purpose, num_classes=self.purpose_dim).float()

    def encode(self, x_trips, x_lengths, x_purpose, x_num, x_cat):
        x_cond = self.embed_conditionals(x_num, x_cat)
        x_purpose_processed = self.process_purpose(x_purpose)
        
        x = torch.cat([
            x_trips.float(), 
            x_lengths.float(), 
            x_purpose_processed, 
            x_cond
        ], dim=1)
        
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, x_num, x_cat):
        x_cond = self.embed_conditionals(x_num, x_cat)
        x = torch.cat([z, x_cond], dim=1)
        h = self.decoder(x)

        trips_logits = [fc(h) for fc in self.fc_trip_logits]
        
        if self.length_distribution == 'gamma':
            # Gamma: shape (alpha) and rate (beta) parameters
            param1s = [F.softplus(fc(h)) + self.min_param for fc in self.fc_param1]  # alpha
            param2s = [F.softplus(fc(h)) + self.min_param for fc in self.fc_param2]  # beta
        elif self.length_distribution == 'beta':
            # Beta: concentration parameters
            param1s = [F.softplus(fc(h)) + self.min_param for fc in self.fc_param1]  # alpha
            param2s = [F.softplus(fc(h)) + self.min_param for fc in self.fc_param2]  # beta
        elif self.length_distribution == 'lognormal':
            # LogNormal: mean and std of underlying normal
            param1s = [fc(h) for fc in self.fc_mu_len]     # mu (can be negative)
            param2s = [F.softplus(fc(h)) + self.min_param for fc in self.fc_sigma_len]  # sigma
        
        purpose_logits = self.fc_purpose(h)
        
        return trips_logits, param1s, param2s, purpose_logits

    def forward(self, x_trips, x_lengths, x_purpose, x_num, x_cat):
        mu, logvar = self.encode(x_trips, x_lengths, x_purpose, x_num, x_cat)
        z = self.reparameterize(mu, logvar)
        trips_logits, param1s, param2s, purpose_logits = self.decode(z, x_num, x_cat)
        return trips_logits, param1s, param2s, purpose_logits, mu, logvar


def vae_loss(
    trips_logits, param1s, param2s, purpose_logits,
    target_trips, target_lengths, target_purpose,
    mu, logvar,
    weights=None,
    distribution_type='gamma',
    beta=1.0,
    lambda_len=1.0,
    length_scale=None,
    debug=False
):
    """
    Enhanced loss function with sample weights support.
    
    Args:
        weights: Sample weights tensor of shape (batch_size,) or None for uniform weighting
                 Weights should be pre-normalized across the entire dataset, not per batch
        distribution_type: 'gamma', 'beta', or 'lognormal'
        length_scale: Optional scaling factor for Beta distribution normalization
    """
    batch_size = target_trips.size(0)
    device = target_trips.device
    
    # Handle weights - do NOT normalize per batch to preserve bias correction
    if weights is None:
        weights = torch.ones(batch_size, device=device)
    else:
        weights = weights.to(device)
    
    if debug:
        print(f"\n=== WEIGHTED LOSS DEBUG ({distribution_type.upper()}) ===")
        print(f"Weight stats - Min: {weights.min().item():.4f}, Max: {weights.max().item():.4f}, "
              f"Mean: {weights.mean().item():.4f}")

    # --- Trip loss (weighted) ---
    trip_loss = torch.tensor(0.0, device=device)
    for i, logits in enumerate(trips_logits):
        # Compute per-sample loss
        loss_per_sample = F.cross_entropy(logits, target_trips[:, i], reduction='none')
        # Apply weights and average
        weighted_loss = (loss_per_sample * weights).mean()
        trip_loss += weighted_loss

    # --- Length loss (weighted) ---
    length_loss = torch.tensor(0.0, device=device)
    
    # Create mask for valid entries
    valid_mask = (target_trips > 0) & (target_lengths > 0)
    
    if valid_mask.sum() > 0:
        # Concatenate parameters
        param1s_cat = torch.cat(param1s, dim=1)
        param2s_cat = torch.cat(param2s, dim=1)
        
        try:
            if distribution_type == 'gamma':
                safe_lengths = torch.clamp(target_lengths, min=1e-6)
                dist = Gamma(param1s_cat, param2s_cat)
                length_log_probs = dist.log_prob(safe_lengths)
                
            elif distribution_type == 'beta':
                safe_lengths = torch.clamp(target_lengths, min=1e-6, max=1-1e-6)
                dist = Beta(param1s_cat, param2s_cat)
                length_log_probs = dist.log_prob(safe_lengths)
                
            elif distribution_type == 'lognormal':
                safe_lengths = torch.clamp(target_lengths, min=1e-6)
                dist = LogNormal(param1s_cat, param2s_cat)
                length_log_probs = dist.log_prob(safe_lengths)
            
            # Handle non-finite values
            if not torch.isfinite(length_log_probs).all():
                if debug:
                    print(f"Non-finite log_probs detected in {distribution_type}")
                length_log_probs = torch.where(torch.isfinite(length_log_probs),
                                             length_log_probs,
                                             torch.tensor(-1000.0, device=device))
            
            # Apply mask and weights
            masked_log_probs = length_log_probs * valid_mask.float()
            # Expand weights to match shape (batch_size, num_modes)
            weights_expanded = weights.unsqueeze(1).expand_as(masked_log_probs)
            weighted_log_probs = masked_log_probs * weights_expanded
            
            # Compute weighted average (only over valid entries)
            valid_weighted_mask = valid_mask.float() * weights_expanded
            length_loss = -weighted_log_probs.sum() / valid_weighted_mask.sum().clamp(min=1e-8)
            
            if debug:
                print(f"Length loss ({distribution_type}): {length_loss.item():.6f}")
                
        except Exception as e:
            if debug:
                print(f"Exception in {distribution_type} computation: {e}")
            length_loss = torch.tensor(0.0, device=device)
    
    # --- Purpose loss (weighted) ---
    purpose_loss_per_sample = F.cross_entropy(purpose_logits, target_purpose, reduction='none')
    purpose_loss = (purpose_loss_per_sample * weights).mean()
    
    # --- KL divergence (weighted) ---
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = (kl_per_sample * weights).mean()
    
    # --- Total loss ---
    total_loss = trip_loss + lambda_len * length_loss + purpose_loss + beta * kl_loss
    
    if debug:
        print(f"Final weighted losses - Trip: {trip_loss.item():.6f}, Length: {length_loss.item():.6f}, "
              f"Purpose: {purpose_loss.item():.6f}, KL: {kl_loss.item():.6f}, "
              f"Total: {total_loss.item():.6f}")
    
    return total_loss, trip_loss, length_loss, purpose_loss, kl_loss


def train_vae(
    model, train_loader, val_loader, optimizer=None, distribution_type='gamma', 
    epochs=50, beta=1.0, patience=10, device='cpu', debug=False, annealation=False
):
    """
    Train CVAE with sample weights support.
    Expected batch format: (x_trips, x_lengths, x_purpose, x_num, x_cat, weights)
    If your DataLoader doesn't include weights, they'll default to uniform.
    """
    train_losses, val_losses = [], []
    train_trip_losses, train_length_losses, train_purpose_losses, train_kl_losses = [], [], [], []

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    model.to(device)

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        epoch_losses = {
            'total': 0, 'trip': 0, 'length': 0, 'purpose': 0, 'kl': 0
        }
        
        num_batches = len(train_loader)

        for batch_idx, batch_data in enumerate(train_loader):
            # Handle optional weights in batch
            if len(batch_data) == 6:
                x_trips, x_lengths, x_purpose, x_num, x_cat, weights = batch_data
                weights = weights.to(device)
                # Print a note that weights are being used
                if batch_idx == 0 and epoch == 0:
                    print("Using sample weights in training.")
            else:
                x_trips, x_lengths, x_purpose, x_num, x_cat = batch_data
                weights = None
            
            # Move data to device
            x_trips = x_trips.to(device)
            x_lengths = x_lengths.to(device)
            x_purpose = x_purpose.to(device)
            if x_num is not None:
                x_num = x_num.to(device)
            if x_cat is not None:
                x_cat = x_cat.to(device)

            optimizer.zero_grad()

            # Forward pass
            trips_logits, param1s, param2s, purpose_logits, mu, logvar = model(
                x_trips, x_lengths, x_purpose, x_num, x_cat
            )

            # Anneal beta if needed
            if annealation:
                beta_anih = min(beta, 0.1 + 1 - (1 - epoch/6)**3) 
            else:
                beta_anih = beta

            # Compute loss with weights
            loss, trip_loss, length_loss, purpose_loss, kl_loss = vae_loss(
                trips_logits, param1s, param2s, purpose_logits,
                x_trips, x_lengths, x_purpose,
                mu, logvar,
                weights=weights,
                distribution_type=distribution_type,
                beta=beta_anih,
                debug=debug
            )

            # Check for NaN losses
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}")
                continue

            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Track losses
            epoch_losses['total'] += loss.item()
            epoch_losses['trip'] += trip_loss.item()
            epoch_losses['length'] += length_loss.item()
            epoch_losses['purpose'] += purpose_loss.item()
            epoch_losses['kl'] += kl_loss.item()

        # Average losses over batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        train_losses.append(epoch_losses['total'])
        train_trip_losses.append(epoch_losses['trip'])
        train_length_losses.append(epoch_losses['length'])
        train_purpose_losses.append(epoch_losses['purpose'])
        train_kl_losses.append(epoch_losses['kl'])

        # --- Validation ---
        model.eval()
        val_epoch_losses = {'total': 0, 'trip': 0, 'length': 0, 'purpose': 0, 'kl': 0}
        val_num_batches = len(val_loader)

        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 6:
                    x_trips, x_lengths, x_purpose, x_num, x_cat, weights = batch_data
                    weights = weights.to(device)
                else:
                    x_trips, x_lengths, x_purpose, x_num, x_cat = batch_data
                    weights = None

                x_trips    = x_trips.to(device)
                x_lengths  = x_lengths.to(device)
                x_purpose  = x_purpose.to(device)
                if x_num is not None: x_num = x_num.to(device)
                if x_cat is not None: x_cat = x_cat.to(device)

                trips_logits, param1s, param2s, purpose_logits, mu, logvar = model(
                    x_trips, x_lengths, x_purpose, x_num, x_cat
                )

                _, val_trip_loss, val_length_loss, val_purpose_loss, val_kl_loss = vae_loss(
                    trips_logits, param1s, param2s, purpose_logits,
                    x_trips, x_lengths, x_purpose,
                    mu, logvar,
                    weights=weights,
                    distribution_type=distribution_type,
                    beta=beta_anih,
                    debug=debug
                )
                val_loss = val_trip_loss + val_length_loss + val_purpose_loss + val_kl_loss

                if not torch.isnan(val_loss):
                    val_epoch_losses['total']   += val_loss.item()
                    val_epoch_losses['trip']    += val_trip_loss.item()
                    val_epoch_losses['length']  += val_length_loss.item()
                    val_epoch_losses['purpose'] += val_purpose_loss.item()
                    val_epoch_losses['kl']      += val_kl_loss.item()

        # Average over batches
        for key in val_epoch_losses:
            val_epoch_losses[key] /= val_num_batches

        avg_val_loss = val_epoch_losses['total']
        val_losses.append(avg_val_loss)

        # --- Early stopping ---
        if (epoch < 11) & (annealation == True):
            patience_counter = 0  # no early stopping in first 10 epochs
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict()) 
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Logging — now all components match the total
        print(f"Epoch {epoch+1}")
        print(f"  Train - Total: {epoch_losses['total']:.4f}, Trip: {epoch_losses['trip']:.4f}, "
            f"Length: {epoch_losses['length']:.4f}, Purpose: {epoch_losses['purpose']:.4f}, "
            f"KL: {epoch_losses['kl']:.4f}")
        print(f"  Val   - Total: {val_epoch_losses['total']:.4f}, Trip: {val_epoch_losses['trip']:.4f}, "
            f"Length: {val_epoch_losses['length']:.4f}, Purpose: {val_epoch_losses['purpose']:.4f}, "
            f"KL: {val_epoch_losses['kl']:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses, train_trip_losses, train_length_losses, train_purpose_losses, train_kl_losses


# Utility function to auto-select embedding dimensions
def auto_select_embedding_dims(cat_dims, max_embed_dim=50):
    """Automatically select embedding dimensions for categorical features."""
    return [min(max_embed_dim, (dim + 1) // 2) for dim in cat_dims]