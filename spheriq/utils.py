import torch.nn.functional as functional
import math
from PIL import Image
import torch
import numpy as np

FACE_SIZE = 512


def is_notebook():
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None and ipy.__class__.__name__ == 'ZMQInteractiveShell':
            return True
        return False
    except Exception:
        return False


def display_text(text):
    if is_notebook():
        try:
            from IPython.display import display
            from ipywidgets import widgets
            html = widgets.HTML(
                value=f"<div style='font-size:large;'>{text}</div>"
            )
            display(html)
            return
        except ImportError:
            pass
    print(text)


def load_equirectangular_image(path, stereo=False):
    img = Image.open(path)

    if stereo:
        size = max(img.width, img.height)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    else:
        size = min(img.width, img.height)
        img = img.resize((size * 2, size), Image.Resampling.LANCZOS)

    img_np = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(img_np).float() / 255.0


_GRID_CACHE = {}


def project_view(view):
    device = view.device
    device_str = str(device)
    batch_size, _, height, width = view.shape

    cache_key = (device_str, height, width, FACE_SIZE)
    if cache_key in _GRID_CACHE:
        grid = _GRID_CACHE[cache_key]
        if grid.device != device:
            grid = grid.to(device)
    else:
        x = torch.linspace(-1, 1, FACE_SIZE, device=device)
        y = torch.linspace(-1, 1, FACE_SIZE, device=device)
        x, y = torch.meshgrid(x, y, indexing="ij")

        xyz = torch.stack([
            torch.stack([torch.ones_like(x), -x, -y]),  # +X (right)
            torch.stack([-torch.ones_like(x), -x, y]),  # -X (left)
            torch.stack([y, torch.ones_like(x), x]),  # +Y (top)
            torch.stack([y, -torch.ones_like(x), -x]),  # -Y (bottom)
            torch.stack([y, -x, torch.ones_like(x)]),  # +Z (front)
            torch.stack([-y, -x, -torch.ones_like(x)]),  # -Z (back)
        ])

        xyz = functional.normalize(xyz, dim=1)

        theta = torch.atan2(xyz[:, 0], xyz[:, 2])  # azimuth
        phi = torch.asin(torch.clamp(xyz[:, 1], -1, 1))  # elevation

        u = (theta / math.pi + 1) * 0.5 * width
        v = (-phi / (math.pi / 2) + 1) * 0.5 * height

        grid = torch.stack([u, v], dim=-1)
        grid[..., 0] = grid[..., 0] / (width - 1) * 2 - 1
        grid[..., 1] = grid[..., 1] / (height - 1) * 2 - 1
        _GRID_CACHE[cache_key] = grid

    # Batched grid_sample: treat the 6 faces as extra batch items so all faces
    # are resampled in a single CUDA/CPU kernel call instead of a Python loop.
    # view: (B, C, H, W) — expand to (6*B, C, H, W) by repeating per face.
    # grid: (6, S, S, 2) — expand to (6*B, S, S, 2) matching the face order.
    num_channels = view.shape[1]
    view_expanded = view.unsqueeze(0).expand(6, -1, -1, -1, -1).reshape(6 * batch_size, num_channels, height, width)
    face_grids = grid.unsqueeze(1).expand(-1, batch_size, -1, -1, -1).reshape(6 * batch_size, FACE_SIZE, FACE_SIZE, 2)

    faces = functional.grid_sample(
        view_expanded,
        face_grids,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )  # (6*B, C, S, S)

    faces = faces.reshape(6, batch_size, num_channels, FACE_SIZE, FACE_SIZE)

    # Return (B, 6, C, S, S)
    return faces.permute(1, 0, 2, 3, 4)


def get_cube_faces(input_val, stereo=False, device="cpu"): # Default changed to cpu
    if isinstance(input_val, str):
        img_tensor = load_equirectangular_image(input_val, stereo=stereo).to(device).unsqueeze(0)
    elif isinstance(input_val, torch.Tensor):
        img_tensor = input_val.to(device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        # If input is PIL, convert to tensor on the specified device (CPU)
        img_np = np.array(input_val).transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_np).float().to(device).unsqueeze(0) / 255.0

    if stereo:
        _, _, h, w = img_tensor.shape
        left_view = img_tensor[:, :, :h // 2, :]
        return project_view(left_view) # project_view uses device from the tensor
    else:
        return project_view(img_tensor)

def equirectangular_to_cube_faces(path, device="cpu"):
    return get_cube_faces(path, stereo=False, device=device)


def rotate_and_project(
    img_tensor: torch.Tensor,
    rotation_matrix: torch.Tensor,
    face_size: int = FACE_SIZE,
) -> torch.Tensor:
    """
    Project an equirectangular image to 6 cubemap faces while applying a
    spherical rotation in a **single** ``grid_sample`` call.

    The naive two-step approach — (1) resample the ERP image into a rotated ERP,
    then (2) resample the rotated ERP into cubemap faces — applies bilinear
    interpolation twice.  Each pass acts as a low-pass filter, blurring high-
    frequency detail.  For an IQA model this is particularly harmful because it
    artificially degrades sharpness before the model even sees the patches.

    Mathematical justification
    --------------------------
    ``project_view`` works by, for each output pixel ``(i,j)`` on face ``f``,
    computing the unit 3-D direction ``d`` that pixel corresponds to, converting
    ``d`` to equirectangular ``(lon, lat)``, and sampling the source ERP there.

    ``SphericalRotation.rotate`` works by, for each output pixel in the *ERP*
    grid, computing its 3-D direction, rotating it by ``R``, and sampling the
    source ERP at the rotated direction.

    Composing: if we want the cube face pixel ``(i,j)`` to show the content that
    would have been at direction ``d`` **before** the rotation was applied, we
    simply apply ``R⁻¹ = Rᵀ`` (rotation matrices are orthogonal) to ``d`` and
    look up that direction in the *original* ERP.  This performs both operations
    with a single interpolation step.

    Args:
        img_tensor:      Float tensor ``(1, C, H, W)`` — the original ERP image.
        rotation_matrix: ``(3, 3)`` rotation matrix ``R`` (same convention as
                         ``SphericalRotation._create_rotation_matrix``).
                         Pass the **forward** rotation; this function applies
                         ``Rᵀ`` internally so the cube face content matches what
                         you would see after rotating the sphere by ``R``.
        face_size:       Output resolution of each cube face (pixels per side).

    Returns:
        ``(1, 6, C, face_size, face_size)`` tensor — same layout as
        ``equirectangular_to_cube_faces``.
    """
    device = img_tensor.device
    batch_size, _, height, width = img_tensor.shape

    # ── Build per-face 3-D direction grids ───────────────────────────────────
    # Identical to project_view: each face pixel maps to a unit direction vector.
    lin = torch.linspace(-1, 1, face_size, device=device)
    x, y = torch.meshgrid(lin, lin, indexing="ij")  # (face_size, face_size) each

    # xyz shape: (6, 3, face_size, face_size) — one direction grid per face
    xyz = torch.stack([
        torch.stack([torch.ones_like(x), -x, -y]),  # +X (right)
        torch.stack([-torch.ones_like(x), -x, y]),  # -X (left)
        torch.stack([y, torch.ones_like(x), x]),  # +Y (top)
        torch.stack([y, -torch.ones_like(x), -x]),  # -Y (bottom)
        torch.stack([y, -x, torch.ones_like(x)]),  # +Z (front)
        torch.stack([-y, -x, -torch.ones_like(x)]),  # -Z (back)
    ])  # (6, 3, S, S)

    xyz = functional.normalize(xyz, dim=1)  # unit vectors

    # ── Apply R⁻¹ = Rᵀ to each direction ────────────────────────────────────
    # We want: source_direction = Rᵀ @ face_direction
    # xyz is (6, 3, S, S); reshape to (6, S*S, 3) for batch matmul.
    R_inv = rotation_matrix.T.to(device=device, dtype=xyz.dtype)  # (3, 3)

    S = face_size
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(6, S * S, 3)  # (6, S², 3)
    # Rotate: (6, S², 3) @ (3, 3)ᵀ  →  (6, S², 3)
    src_flat = xyz_flat @ R_inv.T                              # (6, S², 3)
    src = src_flat.reshape(6, S, S, 3).permute(0, 3, 1, 2)    # (6, 3, S, S)

    # ── Convert rotated directions to ERP (lon, lat) → normalised grid coords ─
    src_x, src_y, src_z = src[:, 0], src[:, 1], src[:, 2]

    # atan2 convention matches project_view: theta = atan2(x, z)
    theta = torch.atan2(src_x, src_z)                     # azimuth  ∈ (-π, π]
    phi   = torch.asin(src_y.clamp(-1.0, 1.0))            # elevation ∈ (-π/2, π/2)

    # Map to pixel coordinates then to grid_sample's [-1, 1] convention
    u = (theta / math.pi + 1.0) * 0.5 * width             # [0, W]
    v = (-phi / (math.pi / 2) + 1.0) * 0.5 * height       # [0, H]

    grid_u = u / (width  - 1) * 2.0 - 1.0                 # [-1, 1]
    grid_v = v / (height - 1) * 2.0 - 1.0                 # [-1, 1]

    # grid_sample expects (B, H_out, W_out, 2) with last dim = (x, y) = (u, v)
    grid = torch.stack([grid_u, grid_v], dim=-1)           # (6, S, S, 2)

    # ── Single batched grid_sample: original ERP → all 6 faces ──────────────
    # grid: (6, S, S, 2) — face-specific sampling coords.
    # img_tensor: (B, C, H, W).
    # Expand both to (6*B, ...) so all faces and all batch items are processed
    # in one kernel call.
    num_channels = img_tensor.shape[1]
    # img_tensor: (B, C, H, W) → (1, B, C, H, W) → (6, B, C, H, W) → (6*B, C, H, W)
    img_expanded = img_tensor.unsqueeze(0).expand(6, -1, -1, -1, -1).reshape(
        6 * batch_size, num_channels, height, width
    )
    # grid: (6, S, S, 2) → (6, 1, S, S, 2) → (6, B, S, S, 2) → (6*B, S, S, 2)
    face_grids = grid.unsqueeze(1).expand(-1, batch_size, -1, -1, -1).reshape(
        6 * batch_size, S, S, 2
    )

    faces_batched = functional.grid_sample(
        img_expanded,
        face_grids,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )  # (6*B, C, S, S)

    # Reshape to (6, B, C, S, S) then permute to (B, 6, C, S, S).
    faces_batched = faces_batched.reshape(6, batch_size, num_channels, S, S)
    return faces_batched.permute(1, 0, 2, 3, 4)  # (B, 6, C, S, S)

def collate_fn(batch):
    """Pads variable-length patch sequences for batching.

    Padding convention:
    - patches:     zero tensor (black patch — CNN sees no signal)
    - spatial_pos / scale_pos / face_ids: 0  (a real position, but the
      attention mask zeroes out any contribution from padding tokens, so
      the embedding lookup value does not matter for correctness)
    - masks:       0.0 for padding slots, 1.0 for real patches
    """
    max_len = max(item['patches'].shape[0] for item in batch)
    patches_list, spatial_list, scale_list, face_list, mask_list, scores = [], [], [], [], [], []

    for item in batch:
        n = item['patches'].shape[0]
        pad = max_len - n
        patches_list.append(torch.cat([
            item['patches'],
            torch.zeros(pad, *item['patches'].shape[1:])  # black patches for padding
        ]))
        spatial_list.append(torch.cat([item['spatial_pos'], torch.zeros(pad, dtype=torch.long)]))
        scale_list.append(torch.cat([item['scale_pos'], torch.zeros(pad, dtype=torch.long)]))
        face_list.append(torch.cat([item['face_ids'], torch.zeros(pad, dtype=torch.long)]))
        
        if 'patch_masks' in item:
            mask_list.append(torch.cat([
                item['patch_masks'],
                torch.zeros(pad, dtype=torch.float32)
            ]))
        else:
            mask_list.append(torch.cat([
                torch.ones(n, dtype=torch.float32),
                torch.zeros(pad, dtype=torch.float32)
            ]))
        scores.append(item['score'])
        
    out = {
        'patches':     torch.stack(patches_list),
        'spatial_pos': torch.stack(spatial_list),
        'scale_pos':   torch.stack(scale_list),
        'face_ids':    torch.stack(face_list),
        'masks':       torch.stack(mask_list),
        'score':       torch.stack(scores),
    }
    
    if 'scene_id' in batch[0]:
        out['scene_id'] = torch.stack([item['scene_id'] for item in batch])
        
    return out