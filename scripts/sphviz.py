"""3D RoPE spatial geometry visualization — ported to spheriq-v9."""

import os
import math
from itertools import product, combinations

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageChops, ImageOps

from spheriq.musiq_arch import MUSIQ

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'sphviz_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_geodesic(ax, p1, p2, color='black'):
    t = np.linspace(0, 1, 20)
    omega = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    if omega < 1e-5:
        return
    arc = (
        (np.sin((1 - t) * omega) / np.sin(omega))[:, None] * p1
        + (np.sin(t * omega) / np.sin(omega))[:, None] * p2
    )
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, linewidth=2, linestyle='--')


def visualize_spherical_geometry(viz_grid=16, freq_scale=None, ckpt_path=None, use_window=True):
    # LaTeX fonts
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        })
    except Exception as e:
        print(f"LaTeX not available, falling back: {e}")

    plt.style.use('default')
    print("Initializing MUSIQ (spheriq-v9) with Spherical Coordinates...")

    model = MUSIQ(
        num_faces=6,
        longer_side_lengths=[512],
        pretrained=False,
        use_spherical_coords=True,
        spatial_pos_grid_size=viz_grid,
        use_face_emb=False,
    )

    n_patches = viz_grid * viz_grid
    points_per_face = n_patches

    # Build flat spatial indices for all 6 faces
    rows_per_face = torch.arange(viz_grid).repeat_interleave(viz_grid)   # (n_patches,)
    cols_per_face = torch.arange(viz_grid).repeat(viz_grid)              # (n_patches,)
    spatial_pos_per_face = rows_per_face * viz_grid + cols_per_face      # (n_patches,)

    all_spatial_pos = spatial_pos_per_face.repeat(6).unsqueeze(0)        # (1, 6*n_patches)
    face_ids = torch.arange(6).repeat_interleave(n_patches).unsqueeze(0) # (1, 6*n_patches)

    # v9 _get_rope_coords(inputs_spatial_positions, face_ids)
    coords = model.transformer_encoder._get_rope_coords(
        all_spatial_pos, face_ids
    )[0, 6:].cpu().numpy()  # skip 6 CLS tokens => (6*n_patches, 3)

    colors = ['red', 'deepskyblue', 'blue', 'darkorange', 'limegreen', 'darkviolet']
    face_names = ['Right (+X)', 'Left (-X)', 'Bottom (-Y)', 'Top (+Y)', 'Front (+Z)', 'Back (-Z)']

    # Reference patches
    idx_front = points_per_face * 4 + (viz_grid // 2 * viz_grid) + (viz_grid - 2)
    idx_right = points_per_face * 0 + (viz_grid // 2 * viz_grid) + 1

    def get_2d_coord(idx):
        face_idx = idx // points_per_face
        local_idx = idx % points_per_face
        r = local_idx // viz_grid
        c = local_idx % viz_grid
        face_grid = {
            0: (1, 2), 1: (1, 0), 2: (2, 1), 3: (0, 1), 4: (1, 1), 5: (1, 3),
        }
        row, col = face_grid[face_idx]
        spacing = 0.5
        start_x = col * (viz_grid + spacing)
        start_y = (2 - row) * (viz_grid + spacing)
        x = start_x + c + 0.5
        y = start_y + r + 0.5
        return x, y

    x_f_2d, y_f_2d = get_2d_coord(idx_front)
    x_r_2d, y_r_2d = get_2d_coord(idx_right)
    p1, p2 = coords[idx_front], coords[idx_right]

    # ── Part 1: 2D Sequence View ──────────────────────────────────────────────
    print("Generating Part 1: 2D Sequence View...")
    fig1 = plt.figure(figsize=(16, 2.5))
    ax2d = fig1.add_subplot(111)
    fig1.patch.set_facecolor('white')
    ax2d.set_title(
        r"Flattened 1D Patch Sequence (Topology-Agnostic)",
        fontsize=16, fontweight='bold', y=1.05,
    )

    samples_per_color = 10
    total_sampled = samples_per_color * 6
    cols = 24
    patch_size = 1.0
    gap_x = 0.2
    gap_y = 0.8

    seq_colors = []
    for face_idx in range(6):
        seq_colors.extend([colors[face_idx]] * samples_per_color)

    for i in range(total_sampled):
        row = i // cols
        col = i % cols
        x = col * (patch_size + gap_x)
        y = -row * (patch_size + gap_y)
        rect = patches.Rectangle(
            (x, y), patch_size, patch_size,
            facecolor=seq_colors[i], edgecolor='black', linewidth=1.2, alpha=0.9,
        )
        ax2d.add_patch(rect)
        if i == 5:
            ax2d.scatter(x + patch_size / 2, y + patch_size / 2,
                         color='darkred', s=150, edgecolors='white', zorder=100)
            ax2d.text(x + patch_size / 2, y - 0.5, r'$p_j$',
                      color='black', fontsize=16, ha='center', fontweight='bold')
        if i == 45:
            ax2d.scatter(x + patch_size / 2, y + patch_size / 2,
                         color='red', s=150, edgecolors='white', zorder=100)
            ax2d.text(x + patch_size / 2, y - 0.5, r'$p_i$',
                      color='black', fontsize=16, ha='center', fontweight='bold')

    ax2d.set_xlim(-1, cols * (patch_size + gap_x) + 1)
    ax2d.set_ylim(-4.0, 1.5)
    ax2d.set_aspect('equal')
    ax2d.axis('off')
    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, 'part1_2d.png'), dpi=600, bbox_inches='tight')
    plt.close(fig1)

    # ── Part 2: 3D Spherical View ─────────────────────────────────────────────
    print("Generating Part 2: 3D Spherical View...")
    fig2 = plt.figure(figsize=(8, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    fig2.patch.set_facecolor('white')
    ax3d.set_facecolor('white')
    ax3d.set_axis_off()
    ax3d.view_init(elev=25, azim=50)

    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    ax3d.plot_surface(x_sphere, y_sphere, z_sphere, color="k", alpha=0.03, zorder=0)

    r_cube = [-1, 1]
    for s, e in combinations(np.array(list(product(r_cube, r_cube, r_cube))), 2):
        if np.sum(np.abs(s - e)) == r_cube[1] - r_cube[0]:
            ax3d.plot3D(*zip(s, e), color="gray", alpha=0.4, linestyle=':')

    coords_unit = coords / np.linalg.norm(coords, axis=-1, keepdims=True)
    p1_unit = coords_unit[idx_front]
    p2_unit = coords_unit[idx_right]

    for face_idx in range(6):
        start = face_idx * points_per_face
        end = start + points_per_face
        face_points = coords_unit[start:end]
        X = face_points[:, 0].reshape(viz_grid, viz_grid)
        Y = face_points[:, 1].reshape(viz_grid, viz_grid)
        Z = face_points[:, 2].reshape(viz_grid, viz_grid)
        center = face_points.mean(axis=0)
        is_front = center.sum() > 0
        face_alpha = 0.5 if is_front else 0.1
        ax3d.plot_wireframe(X, Y, Z, color=colors[face_idx], alpha=face_alpha, linewidth=0.8)
        ax3d.plot_surface(X, Y, Z, color=colors[face_idx], alpha=0.15)

    draw_geodesic(ax3d, p1_unit, p2_unit, color='crimson')
    ax3d.scatter(*p1_unit, color='red', s=60, edgecolors='black', zorder=100)
    ax3d.scatter(*p2_unit, color='darkred', s=60, edgecolors='black', zorder=100)
    mid_point = (p1_unit + p2_unit) / 2
    mid_point = mid_point / np.linalg.norm(mid_point) * 1.35
    ax3d.text(
        mid_point[0], mid_point[1], mid_point[2],
        r'$\mathcal{d}_g(p_i, p_j)$', color='crimson',
        fontweight='bold', fontsize=14,
    )
    ax3d.set_title(r"Restored 3D Spherical Topology", fontsize=16, fontweight='bold', y=0.95)
    limit = 1.35
    ax3d.set_xlim([-limit, limit])
    ax3d.set_ylim([-limit, limit])
    ax3d.set_zlim([-limit, limit])
    ax3d.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'part2_3d.png'), dpi=600, bbox_inches='tight')
    plt.close(fig2)

    # ── Part 3: RoPE Correlation Heatmap ──────────────────────────────────────
    print("Generating Part 3: RoPE Correlation Heatmap...")
    fig3 = plt.figure(figsize=(8, 6))
    ax_heat = fig3.add_subplot(111)
    fig3.patch.set_facecolor('white')
    ax_heat.set_title(r"Implicit 3D RoPE Correlation from $p_i$", fontsize=16, fontweight='bold')

    head_dim = model.hidden_size // model.transformer_encoder.transformer[0].attention.num_heads
    base_dim  = (head_dim // 6) * 2
    half_dim  = base_dim // 2
    print(f"half_dim: {half_dim}")

    # Determine frequency scale factor.
    # Standard RoPE (base=10000) produces frequencies where even the highest
    # component completes < 0.32 cycles across the sphere, making correlation
    # near-uniform.  We scale the frequencies so that the highest component
    # completes `target_cycles` full cycles across the [-1, 1] coordinate range.
    # max_phase = 2.0 * inv_freq[0] * freq_scale  (coord diff of 2.0)
    # target: max_phase / (2*pi) = target_cycles
    # freq_scale = target_cycles * pi / inv_freq[0] = target_cycles * pi
    if freq_scale is None:
        freq_scale = 4.0 * np.pi  # ~4 cycles across the sphere
    print(f"freq_scale: {freq_scale:.2f}")

    # Try loading a trained checkpoint to visualize the actual learned positional correlation.
    if ckpt_path is None:
        ckpt_path = os.path.join(REPO, 'cviq_fold3_best_checkpoint.pth')
    loaded_learned = False

    if os.path.exists(ckpt_path):
        try:
            print(f"Loading trained checkpoint '{ckpt_path}' for learned correlation...")
            _ckpt_full = torch.load(ckpt_path, map_location='cpu')
            state_dict = _ckpt_full.get('model_state_dict', _ckpt_full.get('ema_state_dict', _ckpt_full))

            num_heads = model.transformer_encoder.transformer[0].attention.num_heads
            head_dim_total = model.hidden_size // num_heads
            print(f"  num_heads: {num_heads}, head_dim: {head_dim_total}")

            # Aggregate RoPE correlation across all heads in Layer 1
            rope_corr = np.zeros(coords.shape[0])
            layer = 1
            W_q = state_dict[f'transformer_encoder.transformer.{layer}.attention.query.weight'].numpy()
            W_k = state_dict[f'transformer_encoder.transformer.{layer}.attention.key.weight'].numpy()

            inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_dim) / half_dim))
            inv_freq *= freq_scale

            for head in range(num_heads):
                W_q_head = W_q[head * head_dim_total : (head + 1) * head_dim_total]
                W_k_head = W_k[head * head_dim_total : (head + 1) * head_dim_total]

                alpha = np.zeros((3, half_dim))
                beta  = np.zeros((3, half_dim))
                for axis in range(3):
                    axis_offset = axis * 32
                    for m in range(half_dim):
                        d1 = axis_offset + 2 * m
                        d2 = axis_offset + 2 * m + 1
                        alpha[axis, m] = np.dot(W_q_head[d1], W_k_head[d1]) + np.dot(W_q_head[d2], W_k_head[d2])
                        beta[axis, m]  = np.dot(W_q_head[d2], W_k_head[d1]) - np.dot(W_q_head[d1], W_k_head[d2])

                if use_window:
                    w = np.hanning(half_dim)
                    for axis in range(3):
                        alpha[axis] *= w
                        beta[axis]  *= w

                head_corr = np.zeros(coords.shape[0])
                for axis in range(3):
                    theta_i = p1[axis] * inv_freq
                    theta_j = coords[:, axis:axis+1] * inv_freq[np.newaxis, :]
                    diff = theta_j - theta_i[np.newaxis, :]
                    axis_corr = np.sum(alpha[axis][np.newaxis, :] * np.cos(diff) + beta[axis][np.newaxis, :] * np.sin(diff), axis=1)
                    head_corr += axis_corr

                head_corr = head_corr / (head_corr[idx_front] + 1e-8)
                head_corr = head_corr - head_corr.mean()
                rope_corr += head_corr

            rope_corr /= num_heads
            loaded_learned = True
            print(f"Successfully computed learned positional correlation — averaged over {num_heads} heads!")
            print(f"  corr range: [{rope_corr.min():.4f}, {rope_corr.max():.4f}]  std={rope_corr.std():.4f}")
        except Exception as e:
            print(f"Error loading learned weights: {e}. Falling back to visual proxy.")

    if not loaded_learned:
        print("Using idealized/visual proxy frequency schedule...")
        inv_freq = np.linspace(0.5, 4.0, half_dim) * freq_scale / (2.0 * np.pi)
        rope_corr = np.zeros(coords.shape[0])
        for axis in range(3):
            theta_i = p1[axis] * inv_freq
            theta_j = coords[:, axis:axis+1] * inv_freq[np.newaxis, :]
            diff = theta_j - theta_i[np.newaxis, :]
            rope_corr += np.mean(np.cos(diff), axis=1)
        rope_corr /= 3.0
        # Normalize to [-1, 1] for consistent colorbar
        rope_corr = rope_corr / np.abs(rope_corr).max()
        print(f"  corr range: [{rope_corr.min():.4f}, {rope_corr.max():.4f}]  std={rope_corr.std():.4f}")

    corr_abs_max = max(abs(rope_corr.min()), abs(rope_corr.max()))
    vmin_global = -corr_abs_max
    vmax_global = corr_abs_max
    cmap = 'RdBu_r'

    spacing = 0.5
    for face_idx in range(6):
        face_grid = {
            0: (1, 2), 1: (1, 0), 2: (2, 1), 3: (0, 1), 4: (1, 1), 5: (1, 3),
        }
        row, col = face_grid[face_idx]
        start_x = col * (viz_grid + spacing)
        start_y = (2 - row) * (viz_grid + spacing)
        start = face_idx * points_per_face
        end = start + points_per_face
        face_corr = rope_corr[start:end].reshape(viz_grid, viz_grid)
        im = ax_heat.imshow(
            face_corr,
            extent=(start_x, start_x + viz_grid, start_y, start_y + viz_grid),
            origin='lower', cmap=cmap, alpha=0.9, vmin=vmin_global, vmax=vmax_global,
        )
        rect = patches.Rectangle(
            (start_x, start_y), viz_grid, viz_grid,
            linewidth=1.5, edgecolor='black', facecolor='none', zorder=50,
        )
        ax_heat.add_patch(rect)

    ax_heat.set_xlim(-1, 4 * (viz_grid + spacing) + 0.5)
    ax_heat.set_ylim(-1, 3 * (viz_grid + spacing) + 0.5)
    ax_heat.set_aspect('equal')
    ax_heat.axis('off')

    ax_heat.scatter(x_f_2d, y_f_2d, color='red', s=80, edgecolors='white', zorder=100)
    ax_heat.scatter(x_r_2d, y_r_2d, color='darkred', s=80, edgecolors='white', zorder=100)
    ax_heat.text(x_f_2d, y_f_2d + 1.5, r'$p_i$',
                 color='white', fontsize=14, ha='center', fontweight='bold')
    ax_heat.text(x_r_2d, y_r_2d + 1.5, r'$p_j$',
                 color='white', fontsize=14, ha='center', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_heat, orientation='horizontal', fraction=0.05, pad=0.08)
    if loaded_learned:
        cbar.set_label(
            r'RoPE Positional Correlation (Layer 1, averaged over heads)',
            fontsize=12,
        )
        tick_val = float(f"{vmax_global:.2f}")
        cbar.set_ticks([-tick_val, -tick_val / 2, 0, tick_val / 2, tick_val])
    else:
        cbar.set_label(r'RoPE Positional Correlation (low-freq)', fontsize=12)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, 'part3_heatmap.png'), dpi=600, bbox_inches='tight')
    plt.close(fig3)

    # ── Join into combined layout ─────────────────────────────────────────────
    print("Joining images into combined layout...")

    def trim_white(im):
        bg = Image.new(im.mode, im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            cropped = im.crop(bbox)
            return ImageOps.expand(cropped, border=100, fill='white')
        return im

    image_files = [
        os.path.join(OUTPUT_DIR, 'part1_2d.png'),
        os.path.join(OUTPUT_DIR, 'part2_3d.png'),
        os.path.join(OUTPUT_DIR, 'part3_heatmap.png'),
    ]
    images = [trim_white(Image.open(f).convert('RGB')) for f in image_files]

    top_img = images[0]
    bot_left_img = images[1]
    bot_right_img = images[2]

    bottom_width = bot_left_img.width + bot_right_img.width
    bottom_height = max(bot_left_img.height, bot_right_img.height)

    if top_img.width != bottom_width:
        new_height = int(top_img.height * (bottom_width / top_img.width))
        top_img = top_img.resize((bottom_width, new_height), Image.Resampling.LANCZOS)

    final_width = max(top_img.width, bottom_width)
    final_height = top_img.height + bottom_height

    joined = Image.new('RGB', (final_width, final_height), color='white')
    x_top = (final_width - top_img.width) // 2
    joined.paste(top_img, (x_top, 0))
    x_bottom_start = (final_width - bottom_width) // 2
    y_bottom_start = top_img.height
    joined.paste(bot_left_img, (x_bottom_start, y_bottom_start))
    joined.paste(bot_right_img, (x_bottom_start + bot_left_img.width, y_bottom_start))

    out_path = os.path.join(OUTPUT_DIR, '3D_mapping_combined.png')
    joined.save(out_path)
    print(f"Successfully saved final joined image to {out_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='3D RoPE geometry visualization')
    p.add_argument('--viz-grid', type=int, default=16, help='grid size per face')
    p.add_argument('--freq-scale', type=float, default=None,
                   help='frequency scale factor (default: 2.0, giving one half-cycle from ref to antipode)')
    p.add_argument('--checkpoint', default=None,
                   help='path to checkpoint for learned correlation (default: cviq_fold3)')
    p.add_argument('--no-window', action='store_true',
                   help='disable Hann window (may produce ringing artifacts)')
    args = p.parse_args()
    visualize_spherical_geometry(viz_grid=args.viz_grid, freq_scale=args.freq_scale,
                                  ckpt_path=args.checkpoint, use_window=not args.no_window)
