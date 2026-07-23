r"""MUSIQ model.

Reference:
        Ke, Junjie, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang.
        "Musiq: Multi-scale image quality transformer." In Proceedings of the
        IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5148-5157. 2021.

Ref url: https://github.com/google-research/google-research/tree/master/musiq
Re-implemented by: Chaofeng Chen (https://github.com/chaofengc)

"""

from spheriq.utils import display_text
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spheriq.arch_util import load_pretrained_network
from pyiqa.matlab_utils import ExactPadding2d, exact_padding_2d
from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.data.multiscale_trans_util import get_multiscale_patches
from pyiqa.archs.arch_util import get_url_from_name

default_model_urls = {
    'ava': get_url_from_name('musiq_ava_ckpt-e8d3f067.pth'),
    'koniq10k': get_url_from_name('musiq_koniq_ckpt-e95806b9.pth'),
    'spaq': get_url_from_name('musiq_spaq_ckpt-358bb6af.pth'),
    'paq2piq': get_url_from_name('musiq_paq2piq_ckpt-364c0c84.pth'),
    'imagenet_pretrain': get_url_from_name('musiq_imagenet_pretrain-51d9b0a5.pth'),
}


class StdConv(nn.Conv2d):
    """
    Reference: https://github.com/joe-siyuan-qiao/WeightStandardization
    """

    def forward(self, x):
        # implement same padding
        x = exact_padding_2d(x, self.kernel_size, self.stride, mode='same')
        weight = self.weight
        weight = weight - weight.mean((1, 2, 3), keepdim=True)
        weight = weight / (weight.std((1, 2, 3), keepdim=True) + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super().__init__()

        width = inplanes

        self.conv1 = StdConv(inplanes, width, 1, 1, bias=False)
        self.gn1 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv2 = StdConv(width, width, 3, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, width, eps=1e-4)
        self.conv3 = StdConv(width, outplanes, 1, 1, bias=False)
        self.gn3 = nn.GroupNorm(32, outplanes, eps=1e-4)

        self.relu = nn.ReLU(True)

        self.needs_projection = inplanes != outplanes or stride != 1
        if self.needs_projection:
            self.conv_proj = StdConv(inplanes, outplanes, 1, stride, bias=False)
            self.gn_proj = nn.GroupNorm(32, outplanes, eps=1e-4)

    def forward(self, x):
        identity = x
        if self.needs_projection:
            identity = self.gn_proj(self.conv_proj(identity))

        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        out = self.relu(x + identity)

        return out


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, attn_drop=0.0, out_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def _apply_3d_rope(self, q, k, coords, base=10000.0):
        """
        Correct 3D RoPE: independent per-axis frequency pairs.
        q, k: (B, num_heads, N, head_dim)
        coords: (B, N, 3) continuous values in [-1, 1]
        """
        B, H, N, head_dim = q.shape

        # Divide head_dim into 3 equal blocks (one per spatial axis).
        # Each block has (rope_dim_per_axis) dimensions = half_dim * 2.
        # We need rope_dim_per_axis to be even, so floor to nearest even.
        base_dim   = (head_dim // 6) * 2          # even, per-axis rope width
        rope_dim   = base_dim * 3                  # total rotated dims
        half_dim   = base_dim // 2                 # half of one axis block

        if rope_dim == 0:
            return q, k  # head_dim too small to apply RoPE

        # Frequency schedule: inv_freq has shape (half_dim,)
        inv_freq = 1.0 / (base ** (
            torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / half_dim
        ))

        # coords: (B, N, 3) — split per axis
        cx = coords[:, :, 0]   # (B, N)
        cy = coords[:, :, 1]
        cz = coords[:, :, 2]

        # Outer product: (B, N, half_dim) for each axis
        # Each row i gives [θ_0, θ_1, ..., θ_{half_dim-1}] for that token
        def make_angles(c):
            return c.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)  # (B, N, half_dim)

        # Build cos/sin for each axis: shape (B, N, base_dim)
        # We duplicate angles so that dims [0..half_dim) and [half_dim..base_dim)
        # share the same angle, enabling the rotate_half trick.
        def axis_emb(c):
            angles = make_angles(c)                          # (B, N, half_dim)
            emb    = torch.cat([angles, angles], dim=-1)     # (B, N, base_dim)
            return emb.cos(), emb.sin()

        cos_x, sin_x = axis_emb(cx)
        cos_y, sin_y = axis_emb(cy)
        cos_z, sin_z = axis_emb(cz)

        # Concatenate across axes: (B, N, rope_dim)
        cos_all = torch.cat([cos_x, cos_y, cos_z], dim=-1)  # (B, N, rope_dim)
        sin_all = torch.cat([sin_x, sin_y, sin_z], dim=-1)

        # Expand head dimension: (B, 1, N, rope_dim) for broadcasting over heads
        cos_all = cos_all.unsqueeze(1)   # (B, 1, N, rope_dim)
        sin_all = sin_all.unsqueeze(1)

        def rotate_half_block(x, block_size):
            """Apply rotate_half independently within each block of size block_size."""
            # x: (..., n_blocks * block_size)
            *leading, d = x.shape
            x_blocks = x.reshape(*leading, d // block_size, block_size)
            half = block_size // 2
            x1, x2 = x_blocks[..., :half], x_blocks[..., half:]
            rotated = torch.cat([-x2, x1], dim=-1)
            return rotated.reshape(*leading, d)

        # Split off the non-rotated remainder (if head_dim > rope_dim)
        q_rot,  q_pass  = q[..., :rope_dim],  q[..., rope_dim:]
        k_rot,  k_pass  = k[..., :rope_dim],  k[..., rope_dim:]

        q_rot = q_rot * cos_all + rotate_half_block(q_rot, base_dim) * sin_all
        k_rot = k_rot * cos_all + rotate_half_block(k_rot, base_dim) * sin_all

        return (torch.cat([q_rot, q_pass], dim=-1),
                torch.cat([k_rot, k_pass], dim=-1))

    def forward(self, x, mask=None, rope_coords=None):
        B, N, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if rope_coords is not None:
            q, k = self._apply_3d_rope(q, k, rope_coords)

        # Build an additive attention bias from the padding mask so that
        # F.scaled_dot_product_attention (Flash Attention) can be used.
        # SDPA expects attn_mask with shape (B, heads, N, N) or (B, 1, N, N)
        # and treats it as an additive bias (so 0 = attend, -inf = ignore).
        attn_bias = None
        if mask is not None:
            # mask: (B, N) — 1 for real tokens, 0 for padding
            # Only mask the key axis. Masking the query axis entirely with -inf 
            # causes softmax(-inf) = NaN, which propagates.
            mask_w = mask.reshape(B, 1, 1, N)   # key axis

            # Convert to additive bias: padding key positions → -inf (or -1e4 for fp16)
            attn_bias = torch.zeros(B, 1, N, N, device=x.device, dtype=q.dtype)
            fill_val = torch.finfo(q.dtype).min
            # We expand mask_w to (B, 1, N, N) for the masked_fill
            attn_bias = attn_bias.masked_fill(mask_w.expand(B, 1, N, N) == 0, fill_val)

        # F.scaled_dot_product_attention uses Flash Attention when available,
        # avoiding O(N²) memory allocation and significantly increasing throughput.
        # dropout_p is only applied during training.
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_dim,
        num_heads,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = MultiHeadAttention(
            dim, num_heads, bias=True, attn_drop=attn_drop
        )

        # UPGRADE: Add Stochastic Depth to prevent memorization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, inputs_masks, rope_coords=None):
        x = x + self.drop_path(self.attention(self.norm1(x), inputs_masks, rope_coords))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AddSphericalPositionEmbs(nn.Module):
    """
    Discrete grid-based embeddings.
    Table size = num_faces * num_scales * grid_size², matching the global index formula:
        global = face_id * (num_scales * grid_size²) + scale_pos * grid_size² + local_spatial_pos
    """

    def __init__(self, spatial_pos_grid_size, dim, num_faces=6, num_scales=2):
        super().__init__()
        self.num_faces = num_faces
        self.num_scales = num_scales
        self.spatial_pos_grid_size = spatial_pos_grid_size

        # Total positions across all faces and scales.
        # The global index formula used in MUSIQ.forward is:
        #   face_id * (num_scales * grid²) + scale_pos * grid² + local_spatial_pos
        # so the table must have exactly num_faces * num_scales * grid² rows.
        total_positions = num_faces * num_scales * spatial_pos_grid_size * spatial_pos_grid_size
        self.position_emb = nn.parameter.Parameter(
            torch.randn(total_positions, dim)
        )
        nn.init.normal_(self.position_emb, std=0.02)


    def forward(self, inputs, inputs_positions):
        """
        Args:
            inputs: (B, seq_len_total, dim)
            inputs_positions: (B, seq_len_total) - position indices including face offset
        """
        # Clamp to valid range
        max_idx = self.position_emb.shape[0] - 1
        safe_positions = torch.clamp(inputs_positions.long(), 0, max_idx)

        return inputs + self.position_emb[safe_positions]


class AddFaceEmbs(nn.Module):
    """Adds learnable face embeddings (0-5 for the 6 cubemap faces)."""

    def __init__(self, num_faces, dim):
        super().__init__()
        self.face_emb = nn.parameter.Parameter(torch.randn(num_faces, dim))
        nn.init.normal_(self.face_emb, std=0.02)

    def forward(self, inputs, face_ids):
        """
        Args:
            inputs: (B, seq_len_total, dim)
            face_ids: (B, seq_len_total) - which face each patch belongs to
        """
        return inputs + self.face_emb[face_ids.long()]


class AddScaleEmbs(nn.Module):
    """Adds learnable scale embeddings to the inputs."""

    def __init__(self, num_scales, dim):
        super().__init__()
        self.scale_emb = nn.parameter.Parameter(torch.randn(num_scales, dim))
        nn.init.normal_(self.scale_emb, std=0.02)

    def forward(self, inputs, inputs_scale_positions):
        return inputs + self.scale_emb[inputs_scale_positions.long()]


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            mlp_dim=1152,
            attention_dropout_rate=0.0,
            dropout_rate=0,
            num_heads=4,
            num_layers=14,
            num_scales=3,
            spatial_pos_grid_size=10,
            use_scale_emb=True,
            use_spherical_coords=False,  # NEW: Use 3D coordinates instead of discrete grid
            num_faces=6,  # NEW: Number of cubemap faces
            use_face_emb=True,  # NEW: Add explicit face embeddings
            drop_path_rate=0.0,
    ):
        super().__init__()
        self.use_scale_emb = use_scale_emb
        self.use_spherical_coords = use_spherical_coords
        self.num_faces = num_faces

        # When using spherical coordinate embeddings the 3D unit vectors are
        # face-specific by construction – face 0 always maps to the +X hemisphere,
        # face 1 to −X, etc.  For patches well inside a face this encodes face
        # identity reliably.  However, patches at face *boundaries* have 3D
        # coordinates that are geometrically close to those on the adjacent face,
        # making the spherical coords an ambiguous face discriminator there.
        #
        # Adding a lightweight explicit face embedding on top provides cheap
        # insurance for boundary patches with only 6 extra embedding vectors.
        # We therefore allow use_face_emb with use_spherical_coords, and only
        # warn (not override) if both are requested together.
        self.use_face_emb = use_face_emb

        if use_face_emb and use_spherical_coords:
            import warnings
            warnings.warn(
                "use_face_emb=True is set together with use_spherical_coords=True. "
                "Interior patches already carry face identity via their 3D unit "
                "vectors, but boundary patches benefit from the explicit embedding. "
                "Keeping use_face_emb=True (6 small learnable vectors). "
                "Pass use_face_emb=False to disable it and silence this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Choose positional embedding strategy
        if use_spherical_coords:
            self.spatial_pos_grid_size = spatial_pos_grid_size
        else:
            self.posembed_input = AddSphericalPositionEmbs(spatial_pos_grid_size, input_dim, num_faces, num_scales)

        # Explicit face embeddings — only for the discrete-grid path (see above)
        if self.use_face_emb:
            self.faceembed_input = AddFaceEmbs(num_faces, input_dim)

        # Scale embeddings are always independent of position and face, so they
        # are kept regardless of which positional embedding strategy is chosen.
        if use_scale_emb:
            self.scaleembed_input = AddScaleEmbs(num_scales, input_dim)
        

        # Multi-CLS tokens: one per face to preserve face-level structure
        self.cls = nn.parameter.Parameter(torch.randn(1, num_faces, input_dim))
        nn.init.normal_(self.cls, std=0.02)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_norm = nn.LayerNorm(input_dim, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]  # stochastic depth decay rule
        self.transformer = nn.ModuleList([
            TransformerBlock(input_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, drop_path=dpr[i])
            for i in range(num_layers)
        ])

    def _get_rope_coords(self, inputs_spatial_positions, face_ids):
        n, seq_len = inputs_spatial_positions.shape
        grid_size = self.spatial_pos_grid_size
        i, j = inputs_spatial_positions // grid_size, inputs_spatial_positions % grid_size
        
        def px(k, g):
            return -1.0 + 2.0 * (k.float() + 0.5) / g
            
        px_i = px(i, grid_size)
        px_j = px(j, grid_size)
        
        patch_coords = torch.zeros(n, seq_len, 3, device=inputs_spatial_positions.device, dtype=torch.float32)
        
        for f in range(6):
            m = (face_ids == f)
            m3 = m.unsqueeze(-1).expand(-1, -1, 3)
            if f == 0:
                vals = torch.stack([torch.ones_like(px_i), -px_i, -px_j], dim=-1)
            elif f == 1:
                vals = torch.stack([-torch.ones_like(px_i), -px_i, px_j], dim=-1)
            elif f == 2:
                vals = torch.stack([px_j, torch.ones_like(px_i), px_i], dim=-1)
            elif f == 3:
                vals = torch.stack([px_j, -torch.ones_like(px_i), -px_i], dim=-1)
            elif f == 4:
                vals = torch.stack([px_j, -px_i, torch.ones_like(px_i)], dim=-1)
            elif f == 5:
                vals = torch.stack([-px_j, -px_i, -torch.ones_like(px_i)], dim=-1)
            patch_coords = torch.where(m3, vals, patch_coords)
        
        patch_coords = F.normalize(patch_coords, p=2, dim=-1)
        
        # CLS tokens: centers of each face
        cls_coords = torch.tensor([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ], device=inputs_spatial_positions.device, dtype=torch.float32)
        cls_coords = F.normalize(cls_coords, p=2, dim=-1)
        cls_coords = cls_coords.unsqueeze(0).expand(n, self.num_faces, 3)
        
        return torch.cat([cls_coords, patch_coords], dim=1)

    def forward(
            self,
            x,
            inputs_spatial_positions,
            inputs_local_spatial_pos,
            inputs_scale_positions,
            inputs_masks,
            face_ids=None  # NEW: face IDs for each patch
    ):
        n, _, c = x.shape

        # Add positional embeddings
        if not self.use_spherical_coords:
            x = self.posembed_input(x, inputs_spatial_positions)

        # Add face embeddings if enabled
        if self.use_face_emb and face_ids is not None:
            x = self.faceembed_input(x, face_ids)

        # Add scale embeddings if enabled
        if self.use_scale_emb:
            x = self.scaleembed_input(x, inputs_scale_positions)

        # Prepend multiple CLS tokens (one per face)
        cls_tokens = self.cls.expand(n, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if inputs_masks is not None:
            cls_mask = torch.ones((n, self.num_faces), device=x.device, dtype=inputs_masks.dtype)
            full_mask = torch.cat([cls_mask, inputs_masks], dim=1)
        else:
            full_mask = None

        x = self.dropout(x)

        rope_coords = None
        if self.use_spherical_coords and face_ids is not None:
            rope_coords = self._get_rope_coords(inputs_local_spatial_pos, face_ids)

        for block in self.transformer:
            if self.training:
                # Gradient checkpointing trades compute for memory during training.
                # It must be disabled in eval mode: (a) no gradients are needed,
                # so recomputation is pure waste, and (b) under torch.no_grad() the
                # recompute pass skips autograd entirely anyway, making it a no-op
                # that still incurs two forward passes worth of compute.
                if not x.requires_grad:
                    x.requires_grad_(True)
                x = torch.utils.checkpoint.checkpoint(
                    block, x, full_mask, rope_coords,
                    use_reentrant=True,
                )
            else:
                # BUG FIX: eval mode must still run transformer blocks.
                # The previous code had no else branch, so during evaluation
                # the entire transformer was silently skipped — producing near-
                # constant outputs and NaN/zero correlations.
                x = block(x, full_mask, rope_coords)
        x = self.encoder_norm(x)
        if full_mask is not None:
            x = x * full_mask.unsqueeze(-1)
        return x


class SafeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        x = self.norm1(src)
        x, _ = self.self_attn(
            x, x, x, 
            key_padding_mask=src_key_padding_mask, 
            attn_mask=attn_bias,
            need_weights=False
        )
        src = src + self.dropout1(x)
        
        x = self.norm2(src)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x)
        return src


class SafeMetaTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            SafeTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        for mod in self.layers:
            if self.training:
                if not src.requires_grad:
                    src.requires_grad_(True)
                src = torch.utils.checkpoint.checkpoint(
                    mod, src, src_key_padding_mask, attn_bias,
                    use_reentrant=True,
                )
            else:
                src = mod(src, src_key_padding_mask=src_key_padding_mask, attn_bias=attn_bias)
        return self.norm(src)


@ARCH_REGISTRY.register()
class MUSIQ(nn.Module):
    def __init__(
            self,
            patch_size=32,
            num_class=1,
            hidden_size=384,
            mlp_dim=1152,
            attention_dropout_rate=0.0,
            dropout_rate=0,
            num_heads=4,
            num_layers=14,
            spatial_pos_grid_size=10,
            use_scale_emb=True,
            use_spherical_coords=False,
            use_face_emb=True,
            num_faces=6,
            pretrained=True,
            pretrained_model_path=None,
            longer_side_lengths=None,  # default set below to avoid mutable default
            max_seq_len_from_original_res=-1,
            drop_path_rate=0.1,
    ):
        super(MUSIQ, self).__init__()
        if longer_side_lengths is None:
            longer_side_lengths = [224, 384, 512]

        # num_scales is derived directly from longer_side_lengths so it is always consistent.
        # get_multiscale_patches produces one scale per entry in longer_side_lengths, plus the
        # original resolution when max_seq_len_from_original_res != 0.
        uses_original_res = (max_seq_len_from_original_res != 0)
        num_scales = len(longer_side_lengths) + (1 if uses_original_res else 0)

        resnet_token_dim = 64
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_faces = num_faces
        self.num_scales = num_scales

        self.data_preprocess_opts = {
            'patch_size': patch_size,
            'patch_stride': patch_size,
            'hse_grid_size': spatial_pos_grid_size,
            'longer_side_lengths': longer_side_lengths,
            'max_seq_len_from_original_res': max_seq_len_from_original_res,
        }

        # Handle pretrained model
        load_pretrained = False
        if pretrained_model_path is None and pretrained:
            url_key = 'ava' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
            load_pretrained = True

        # CNN backbone
        self.conv_root = StdConv(3, resnet_token_dim, 7, 2, bias=False)
        self.gn_root = nn.GroupNorm(32, resnet_token_dim, eps=1e-6)
        self.root_pool = nn.Sequential(
            nn.ReLU(True),
            ExactPadding2d(3, 2, mode='same'),
            nn.MaxPool2d(3, 2),
        )

        token_patch_size = patch_size // 4
        self.block1 = Bottleneck(resnet_token_dim, resnet_token_dim * 4)

        # Embedding layer
        self.embedding = nn.Linear(
            resnet_token_dim * 4 * token_patch_size ** 2, hidden_size
        )

        # Transformer encoder with spherical awareness
        self.transformer_encoder = TransformerEncoder(
            hidden_size,
            mlp_dim,
            attention_dropout_rate,
            dropout_rate,
            num_heads,
            num_layers,
            num_scales,
            spatial_pos_grid_size,
            use_scale_emb,
            use_spherical_coords,
            num_faces,
            use_face_emb,
            drop_path_rate,
        )

        self.head = nn.Linear(hidden_size, num_class)
        # NOTE: self.head provides an auxiliary prediction based on the CLS tokens
        # to ensure the transformer's CLS pathway remains well-trained.

        # ── Learned Face Aggregation (Meta-Transformer) ───────────────────────
        # A small 3-layer Transformer takes the 6 face CLS tokens as input and
        # produces a single global quality score.  This lets the model learn
        # which faces are most diagnostic for quality (e.g. equatorial > polar)
        # rather than fixing that bias in the inductive prior.
        #
        # Design choices:
        #   • 3 SA layers for the 6-token + AGG + patches sequence.
        #   • dim = hidden_size (no projection bottleneck).  The meta-transformer
        #     is where all 6 face quality signals are fused; compressing to
        #     hidden_size // 2 discards representational capacity at the most
        #     critical aggregation step.  With only 7 tokens the parameter
        #     savings (~0.6 M for hidden_size=384) are negligible compared to
        #     the total model size (~32 M), so the tradeoff is not worthwhile.
        #   • A learned [AGG] token (analogous to CLS) gathers information from
        #     all 6 face tokens and is read out by the final linear head.
        meta_dim = hidden_size          # no bottleneck projection
        self.meta_proj   = nn.Identity()  # kept for checkpoint key compatibility
        self.meta_agg_token = nn.Parameter(torch.randn(1, 1, meta_dim) * 0.02)

        self.face_id_embed = nn.Embedding(6, meta_dim)
        nn.init.normal_(self.face_id_embed.weight, std=0.02)

        # Use SafeMetaTransformer to bypass the buggy PyTorch nn.TransformerEncoder
        # fast path which causes identical constant outputs during evaluation 
        # when a float src_key_padding_mask is provided.
        self.meta_transformer = SafeMetaTransformer(
            d_model=meta_dim,
            nhead=max(1, meta_dim // 64),
            dim_feedforward=meta_dim * 4,
            dropout=dropout_rate,
            num_layers=3,
        )
        self.meta_head_mean = nn.Linear(meta_dim, num_class)
        self.meta_head_logvar = nn.Linear(meta_dim, num_class)
        nn.init.zeros_(self.meta_head_logvar.weight)
        nn.init.zeros_(self.meta_head_logvar.bias)

        # ── Equirectangular Distortion Weighting ──────────────────────────────
        # Pre-compute per-face solid-angle weights, normalised so the 6 weights sum to 1.
        # Poles are over-sampled in equirectangular projection and should receive less
        # training signal.
        #
        # The correct weight is derived from the solid-angle integral over each cubemap face:
        #
        #   Ω(face) ∝ ∫∫_{-π/4}^{π/4} cos³θ / (1+tan²u+tan²v) du dv
        #
        # Numerically this yields:
        #   Equatorial faces (+X, -X, +Z, -Z): relative weight ≈ 1.000
        #   Polar faces (+Y, -Y):              relative weight ≈ 0.552
        #
        # The previous approximation sin(φ_center) = sin(45°) ≈ 0.707 sampled only
        # the face centre latitude, over-weighting polar faces by ~28%.
        # Order matches project_view: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z
        from spheriq.constants import _RAW_ERP_WEIGHTS as raw_lat_weights
        self.register_buffer(
            'erp_face_weights',
            raw_lat_weights / raw_lat_weights.sum(),   # normalise to sum = 1
            persistent=False
        )

        # ── Learned Attention Bias for Meta-Transformer ──────────────────────
        # meta_input layout: [AGG(1) | CLS_0..5(6) | patches(n_patches)]
        #
        # Non-patch → non-patch: learnable (7, 7) bias matrix.
        #   Row = query type, Column = key type.
        #   Initialised so AGG→face and CLS→face biases roughly follow the
        #   ERP solid-angle weights (log-scale), but each query-key pair can
        #   learn its own value.
        #
        # Non-patch → patch: per-face column bias (from solid-angle), scaled
        #   per query type by meta_query_bias_scale.
        #
        # Patch → non-patch: zero (patches have no prior toward CLS/AGG).
        #
        # Patch → patch: column-broadcast per-face bias (same as before).
        _erp_w = raw_lat_weights / raw_lat_weights.sum()              # (6,) normalised
        _face_bias_init = torch.log(_erp_w).clamp(min=-2.0)           # (6,)
        _nonpatch_init = torch.zeros(7, 7)
        # AGG→CLS faces: start at log(erp_weight) (recall _face_bias_init is negative)
        _nonpatch_init[0, 1:] = _face_bias_init                       # (7,)
        # CLS→CLS faces: strong self-bias + cross-face based on erp weight
        for i in range(6):
            _nonpatch_init[1 + i, 1 + i] = -0.5                       # self-attention centre
            _nonpatch_init[1 + i, 1:] += _face_bias_init * 0.5        # cross-face bias
        self.nonpatch_attn_bias = nn.Parameter(
            _nonpatch_init + torch.randn(7, 7) * 0.02
        )

        # Per-face column bias for patch tokens (solid-angle based)
        self.register_buffer('meta_patch_bias', _face_bias_init, persistent=True)

        # Per-query-type scaling: each of the 7 non-patch query types can
        # independently amplify or suppress its bias toward all keys.
        self.meta_query_bias_scale = nn.Parameter(torch.ones(7))

        self.grid_size = self.data_preprocess_opts['hse_grid_size']

        # Load pretrained weights with proper handling
        if load_pretrained and pretrained_model_path is not None:
            self._load_pretrained_with_modifications(pretrained_model_path)

    def _load_pretrained_with_modifications(self, model_path):
        """
        Load pretrained weights, skipping incompatible layers (position embeddings).
        """
        from pyiqa.utils.download_util import load_file_from_url

        if model_path.startswith('https://') or model_path.startswith('http://'):
            model_path = load_file_from_url(model_path)

        display_text(f'Loading pretrained model with modifications from {model_path}')

        pretrained_dict = torch.load(
            model_path, map_location=torch.device('cpu'), weights_only=False
        )

        # Clean state dict
        from spheriq.arch_util import clean_state_dict
        pretrained_dict = clean_state_dict(pretrained_dict)

        model_dict = self.state_dict()

        # Filter out incompatible keys (position embeddings are different size)
        filtered_dict = {}
        incompatible_keys = []

        for k, v in pretrained_dict.items():
            new_k = k.replace('encoderblock_', '')
            if new_k in ['head.weight', 'head.bias', 'meta_head.weight', 'meta_head.bias']:
                incompatible_keys.append(k)
                print(f"Skipping {k}: re-initializing for z-score targets")
                continue
            
            if new_k in model_dict:
                if model_dict[new_k].shape == v.shape:
                    filtered_dict[new_k] = v
                elif new_k == 'transformer_encoder.cls' and v.shape[1] == 1:
                    print(f"Replicating {new_k} from 1 to {self.num_faces} tokens")
                    filtered_dict[new_k] = v.repeat(1, self.num_faces, 1)
                elif new_k == 'embedding.weight' and len(v.shape) == 2 and len(model_dict[new_k].shape) == 2:
                    hidden_size = v.shape[0]
                    in_channels = 256
                    old_spatial = int((v.shape[1] / in_channels) ** 0.5)
                    new_spatial = int((model_dict[new_k].shape[1] / in_channels) ** 0.5)
                    
                    if old_spatial ** 2 * in_channels == v.shape[1] and new_spatial ** 2 * in_channels == model_dict[new_k].shape[1]:
                        print(f"Interpolating {new_k} from spatial grid {old_spatial}x{old_spatial} to {new_spatial}x{new_spatial}")
                        v_reshaped = v.view(hidden_size, in_channels, old_spatial, old_spatial)
                        v_interp = torch.nn.functional.interpolate(
                            v_reshaped, size=(new_spatial, new_spatial), mode='bilinear', align_corners=False
                        )
                        filtered_dict[new_k] = v_interp.reshape(hidden_size, -1)
                    else:
                        incompatible_keys.append(k)
                        print(f"Skipping {k}: shape mismatch {v.shape} vs {model_dict[new_k].shape}")
                else:
                    incompatible_keys.append(k)
                    print(f"Skipping {k}: shape mismatch {v.shape} vs {model_dict[new_k].shape}")
            else:
                incompatible_keys.append(k)

        # Load compatible weights
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict, strict=False)

        print(f"Loaded {len(filtered_dict)}/{len(pretrained_dict)} layers from pretrained model")
        if incompatible_keys:
            print(f"Skipped incompatible keys: {incompatible_keys}")

    def load_state_dict(self, state_dict, strict=True):
        """
        Override to transparently handle checkpoints saved before the meta_proj
        bottleneck was removed (i.e. when meta_proj was nn.Linear(H, H//2)).
        
        Also handles migration from scalar meta_head to mean/logvar meta_head,
        and from the old column-only attention bias to the new full (7,7) bias.
        """
        if 'meta_head.weight' in state_dict:
            state_dict['meta_head_mean.weight'] = state_dict.pop('meta_head.weight')
        if 'meta_head.bias' in state_dict:
            state_dict['meta_head_mean.bias'] = state_dict.pop('meta_head.bias')

        stale_keys = {
            k for k in state_dict
            if k.startswith('meta_proj.') or k.startswith('erp_weight_proj.')
            or k == 'face_weight_logits'
            or k == 'meta_attn_cls_bias' or k == 'meta_attn_bias_scale'
        }
        if stale_keys:
            import warnings
            warnings.warn(
                f"Checkpoint contains stale keys {stale_keys}. "
                "These belong to removed layers and will be discarded.",
                UserWarning,
                stacklevel=2,
            )
            state_dict = {k: v for k, v in state_dict.items() if k not in stale_keys}
            
        # Always use strict=False to avoid missing key errors for meta_head_logvar
        # and discarded layers.
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, patches, spatial_pos, scale_pos, face_ids, masks=None, return_aux=True):
        """
        Args:
            patches: (b, n, c, h, w) - The pre-extracted patches from the Dataset
            spatial_pos: (b, n) - Local spatial indices within a face
            scale_pos: (b, n) - Scale indices (0 for 1x, 1 for 0.5x, etc.)
            face_ids: (b, n) - Face indices (0 to 5)
            masks: (b, n) - Optional padding masks
        """
        b, n, c, h, w = patches.shape

        # 1. Compute Global Spatial Indices
        # spatial_pos is in [0, grid_size²) — a local cell within one face at one scale.
        # We must separate by both face AND scale to avoid collisions:
        #   slot layout: face_id * (num_scales * grid_size²) + scale_pos * grid_size² + local_spatial_pos
        grid_size_sq = self.data_preprocess_opts['hse_grid_size'] ** 2
        global_spatial_pos = (
            face_ids * (self.num_scales * grid_size_sq)
            + scale_pos * grid_size_sq
            + spatial_pos
        )

        # 2. CNN Backbone (Root Processing)
        # Flatten batch and sequence for the CNN root
        x = patches.reshape(-1, c, h, w)
        x = self.conv_root(x)
        x = self.gn_root(x)
        x = self.root_pool(x)
        if self.training:
            if not x.requires_grad:
                x.requires_grad_(True)
            x = torch.utils.checkpoint.checkpoint(self.block1, x, use_reentrant=True)
        else:
            x = self.block1(x)
        
        # 3. Embedding
        x = x.permute(0, 2, 3, 1).reshape(b, n, -1)
        x = self.embedding(x)

        # 4. Transformer Encoder
        # We pass both global and local spatial pos
        x = self.transformer_encoder(
            x, 
            global_spatial_pos, 
            spatial_pos,
            scale_pos, 
            masks,
            face_ids
        )

        # 5. Meta-Transformer Face Aggregation
        # Instead of just the 6 face CLS tokens, we allow the [AGG] token to attend 
        # directly to the entire raw patch sequence for fine-grained localization.
        
        # ── Step 1: separate encoder output into CLS tokens and patch tokens ──────
        # transformer_encoder returns (b, num_faces + n_patches, d).
        # The first num_faces positions are the per-face CLS tokens; the rest are patches.
        x_cls     = x[:, :self.num_faces, :]          # (b, 6, d)
        x_patches = x[:, self.num_faces:, :]          # (b, n_patches, d)

        # ── Step 2: build meta sequence from CLS + patch tokens ────────────────────
        # Add face_id_embed to CLS tokens so the meta-transformer knows which face they represent
        cls_face_ids = torch.arange(self.num_faces, device=x.device).expand(b, -1)
        x_cls_meta = x_cls + self.face_id_embed(cls_face_ids)
        x_patches_meta = x_patches + self.face_id_embed(face_ids.long())   # (b, n_patches, d)

        # ── Step 3: build padding mask for meta-transformer ───────────────────────
        # meta_input will be [agg | x_cls_meta | x_patches_meta], length = 1 + 6 + n_patches.
        n_patches = x_patches.shape[1]
        padding_mask = torch.zeros((b, 1 + self.num_faces + n_patches), device=x.device, dtype=torch.bool)
        if masks is not None:
            # True → ignored by MultiheadAttention (inverted from collate_fn convention)
            padding_mask[:, 1 + self.num_faces:] = (masks == 0)
        # padding_mask[:, :1+self.num_faces] stays False: agg and CLS tokens are never masked.

        # ── Step 3.5: build per-sample additive attention bias ────────────────────
        # meta_input layout: [AGG(1) | CLS_0..5(6) | patches(n_patches)]
        #
        # Bias is a full (n_total, n_total) matrix with four quadrants:
        #
        #   nonpatch→nonpatch (7×7) | nonpatch→patch  (7×n_patches)
        #   ─────────────────────────┼─────────────────────────────
        #   patch→nonpatch  (0s)     | patch→patch    (col-bcast)
        #
        # nonpatch→nonpatch: learnable per-query→per-key bias matrix,
        #   scaled per query type by meta_query_bias_scale (7,).
        # nonpatch→patch: per-face column bias (from ERP solid-angle),
        #   scaled per query type by meta_query_bias_scale.
        # patch→nonpatch: zero (patches have no prior bias toward CLS/AGG).
        # patch→patch: column-broadcast per-face bias (same as originally).
        #
        # We reshape to (b * nhead, n_total, n_total) at the end.

        n_total = 1 + self.num_faces + n_patches
        nhead = self.meta_transformer.layers[0].self_attn.num_heads

        # Per-patch, per-sample face bias: face_ids is (b, n_patches)
        patch_bias = self.meta_patch_bias[face_ids.long()]            # (b, n_patches)

        # Top-left: nonpatch → nonpatch, (7, 7), scaled per query type
        np_bias_scaled = self.nonpatch_attn_bias * self.meta_query_bias_scale[:, None]  # (7, 7)

        # Top-right: nonpatch → patch, (b, 7, n_patches), scaled per query type
        np2patch = self.meta_query_bias_scale[None, :, None] * patch_bias[:, None, :]  # (b, 7, n_patches)

        # Bottom-left: patch → nonpatch — zero bias
        patch2np = torch.zeros(b, n_patches, 7, device=x.device, dtype=x.dtype)

        # Bottom-right: patch → patch — column-broadcast per-face bias
        patch2patch = patch_bias[:, :, None].expand(b, n_patches, n_patches)  # (b, n_patches, n_patches)

        # Top-left expanded to batch
        top_left = np_bias_scaled.unsqueeze(0).expand(b, 7, 7)       # (b, 7, 7)

        # Stack quadrants
        top = torch.cat([top_left, np2patch], dim=2)                 # (b, 7, 7+n_patches)
        bottom = torch.cat([patch2np, patch2patch], dim=2)           # (b, n_patches, 7+n_patches)
        attn_bias = torch.cat([top, bottom], dim=1)                  # (b, n_total, n_total)

        # Reshape to (b * nhead, n_total, n_total) for nn.MultiheadAttention
        attn_bias = attn_bias.unsqueeze(1).expand(b, nhead, n_total, n_total)     # (b, nhead, n_total, n_total)
        attn_bias = attn_bias.reshape(b * nhead, n_total, n_total)                # (b*nhead, n_total, n_total)
        attn_bias = attn_bias.contiguous()

        if masks is not None:
            # Fold padding_mask into attn_bias to avoid PyTorch type mismatch warning
            padding_mask_expanded = padding_mask.view(b, 1, 1, n_total).expand(b, nhead, n_total, n_total).reshape(b * nhead, n_total, n_total)
            attn_bias = attn_bias.masked_fill(padding_mask_expanded, torch.finfo(x.dtype).min)

        # ── Step 4: run meta-transformer and read out agg token ───────────────────
        agg        = self.meta_agg_token.expand(b, -1, -1)         # (b, 1, d)
        meta_input = torch.cat([agg, x_cls_meta, x_patches_meta], dim=1) # (b, 1+6+n_patches, d)
        meta_out   = self.meta_transformer(
            meta_input, 
            src_key_padding_mask=None,
            attn_bias=attn_bias,           # NEW
        )
        pooled_out = meta_out[:, 0]                                 # agg token output
        mean       = self.meta_head_mean(pooled_out)                # (b, 1)

        if self.training:
            logvar = self.meta_head_logvar(pooled_out).clamp(-2, 2)
            logits = (mean, logvar)
        else:
            logits = mean

        # ── Step 5 (optional but recommended): use self.head as an auxiliary loss ─
        # x_cls contains per-face quality estimates. Averaging them with ERP weights
        # gives a free auxiliary prediction that keeps the encoder's CLS pathway trained.
        if return_aux:
            face_logits = self.head(x_cls).squeeze(-1)                  # (b, 6)
            aux_pred = face_logits
        else:
            aux_pred = None

        return logits, aux_pred

    @property
    def face_distortion_weights(self) -> torch.Tensor:
        """Return the pre-computed ERP distortion weights (6,) for external use."""
        return self.erp_face_weights