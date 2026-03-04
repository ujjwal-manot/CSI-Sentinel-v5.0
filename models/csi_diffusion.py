import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        h = h.view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)

        return x + h


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        downsample: bool = True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResidualBlock(ch_in, out_channels, time_embedding_dim))

            if use_attention:
                self.attentions.append(AttentionBlock(out_channels))
            else:
                self.attentions.append(nn.Identity())

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states = []

        for block, attn in zip(self.blocks, self.attentions):
            x = block(x, time_emb)
            x = attn(x)
            hidden_states.append(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x, hidden_states


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        upsample: bool = True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.blocks.append(ResidualBlock(ch_in + out_channels, out_channels, time_embedding_dim))

            if use_attention:
                self.attentions.append(AttentionBlock(out_channels))
            else:
                self.attentions.append(nn.Identity())

        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        for block, attn, skip in zip(self.blocks, self.attentions, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_emb)
            x = attn(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int, ...] = (16, 8),
        num_res_blocks: int = 2,
        time_embedding_dim: int = 256,
        num_classes: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )

        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_embedding_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_res = 128

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            downsample = i < len(channel_mults) - 1

            self.down_blocks.append(
                DownBlock(
                    channels[-1], out_ch, time_embedding_dim,
                    num_res_blocks, use_attn, downsample
                )
            )
            channels.append(out_ch)
            if downsample:
                current_res //= 2

        mid_channels = channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_embedding_dim)
        self.mid_attention = AttentionBlock(mid_channels)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_embedding_dim)

        self.up_blocks = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            upsample = i < len(channel_mults) - 1

            self.up_blocks.append(
                UpBlock(
                    channels.pop(), out_ch, time_embedding_dim,
                    num_res_blocks, use_attn, upsample
                )
            )
            if upsample:
                current_res *= 2

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        time_emb = get_timestep_embedding(timesteps, self.conv_in.out_channels)
        time_emb = self.time_mlp(time_emb)

        if class_labels is not None and self.num_classes is not None:
            class_emb = self.class_embedding(class_labels)
            time_emb = time_emb + class_emb

        x = self.conv_in(x)

        skip_connections = []
        for down_block in self.down_blocks:
            x, hiddens = down_block(x, time_emb)
            skip_connections.extend(hiddens)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_block2(x, time_emb)

        for up_block in self.up_blocks:
            num_skips = len(up_block.blocks)
            block_skips = skip_connections[-num_skips:]
            skip_connections = skip_connections[:-num_skips]
            x = up_block(x, time_emb, block_skips)

        x = self.conv_out(x)

        return x


class CSIDiffusion(nn.Module):
    def __init__(
        self,
        unet: Optional[DiffusionUNet] = None,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        super().__init__()

        self.unet = unet or DiffusionUNet()
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos((t / num_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.unet(x_noisy, t, class_labels)

        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        loss = self.p_losses(x, t, class_labels)

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        predicted_noise = self.unet(x, t, class_labels)

        beta = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]

        model_mean = sqrt_recip_alpha * (x - beta * predicted_noise / sqrt_one_minus_alpha)

        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.posterior_variance[t][:, None, None, None]
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        channels: int = 3,
        height: int = 128,
        width: int = 128,
        class_labels: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device

        x = torch.randn(batch_size, channels, height, width, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, class_labels)

        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        batch_size: int,
        channels: int = 3,
        height: int = 128,
        width: int = 128,
        class_labels: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        device = device or next(self.parameters()).device

        step_size = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)

        x = torch.randn(batch_size, channels, height, width, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            predicted_noise = self.unet(x, t_batch, class_labels)

            alpha = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            alpha_ratio = (1 - alpha_prev) / (1 - alpha + 1e-8) * (1 - alpha / (alpha_prev + 1e-8))
            sigma = eta * torch.sqrt(torch.clamp(alpha_ratio, min=0))

            pred_x0 = (x - torch.sqrt(1 - alpha) * predicted_noise) / (torch.sqrt(alpha) + 1e-8)
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=0)) * predicted_noise

            if sigma > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
            else:
                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        return x
