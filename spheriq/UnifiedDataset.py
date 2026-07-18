import torch
import random
import math
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as TF

from spheriq.rotate import SphericalRotation
from pyiqa.data.multiscale_trans_util import get_multiscale_patches
from spheriq.utils import equirectangular_to_cube_faces, rotate_and_project

# ---------------------------------------------------------------------------
# ERP distortion weight per face (solid-angle proxy, normalised to sum = 1)
# Order: 0:+X(right)  1:-X(left)  2:+Y(top)  3:-Y(bottom)  4:+Z(front)  5:-Z(back)
# ---------------------------------------------------------------------------

class SyntheticArtifactAugmentation:
    """
    Randomly applies one of three synthetic distortion types to a patch tensor.

    Motivation: models trained only on the specific distortions in a dataset
    (e.g. JPEG artefacts) fail to generalise to unseen distortion types.
    Randomly injecting Gaussian blur, additive Gaussian noise, or downsampling
    forces the model to learn the *concept* of quality degradation rather than
    memorising dataset-specific noise patterns.

    Args:
        prob (float): Probability that any augmentation is applied at all.
        blur_sigma_range (tuple): (min, max) sigma for Gaussian blur.
        noise_std_range  (tuple): (min, max) std for additive Gaussian noise.
        downsample_range (tuple): (min, max) downscale factor ∈ (0, 1].
    """

    def __init__(
        self,
        prob: float = 0.3,
        blur_sigma_range: tuple = (0.5, 2.0),
        noise_std_range:  tuple = (0.01, 0.08),
        downsample_range: tuple = (0.4, 0.9),
    ):
        self.prob = prob
        self.blur_sigma_range  = blur_sigma_range
        self.noise_std_range   = noise_std_range
        self.downsample_range  = downsample_range

    def sample_params(self):
        """
        Sample artifact parameters without applying.
        Returns:
            dict or None: None means no augmentation.
        """
        if random.random() > self.prob:
            return None
        choice = random.randint(0, 2)
        params = {'choice': choice}
        if choice == 0:
            params['sigma'] = random.uniform(*self.blur_sigma_range)
        elif choice == 1:
            params['std'] = random.uniform(*self.noise_std_range)
        else:
            params['scale'] = random.uniform(*self.downsample_range)
        return params

    def apply_params(self, img: torch.Tensor, params) -> torch.Tensor:
        """
        Apply pre-sampled artifact parameters to an image.

        Args:
            img: Float tensor (C, H, W), arbitrary value range.
            params: Output of sample_params(), or None.
        Returns:
            Distorted tensor of the same shape.
        """
        if params is None:
            return img
        choice = params['choice']
        if choice == 0:
            sigma = params['sigma']
            ks = max(3, int(math.ceil(6 * sigma)) | 1)
            img = F.gaussian_blur(img, kernel_size=[ks, ks], sigma=[sigma, sigma])
        elif choice == 1:
            std = params['std']
            img = img + torch.randn_like(img) * std
            img = img.clamp(-1.0, 1.0)
        else:
            scale = params['scale']
            _, h, w = img.shape
            small = TF.interpolate(
                img.unsqueeze(0),
                size=(max(4, int(h * scale)), max(4, int(w * scale))),
                mode='bicubic',
                align_corners=False,
            )
            img = TF.interpolate(small, size=(h, w), mode='bicubic', align_corners=False).squeeze(0)
            img = img.clamp(-1.0, 1.0)
        return img

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convenience: sample and apply in one call.
        Args:
            img: Float tensor (C, H, W), arbitrary value range.
        Returns:
            Distorted tensor of the same shape.
        """
        return self.apply_params(img, self.sample_params())


class UnifiedODIQADataset(Dataset):
    def __init__(self, image_paths, scores, hse_opts, 
                 augment=True, device='cpu', artifact_aug_prob=0.3, stereo=False, scene_ids=None):
        self.image_paths = image_paths
        self.scores = scores
        self.scene_ids = scene_ids
        self.hse_opts = hse_opts
        self.augment = augment
        self.device = 'cpu'
        self.yaw = None  # For TTA
        self.pitch = None
        self.roll = None
        self.stereo = stereo  # Whether images are stereoscopic ODS (Top-Bottom format)
        
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

        # Full 6-DOF jitter ranges (applied during training only)
        # Pitch ±15°  and  Roll ±10°  simulate realistic HMD orientations while
        # keeping the scene recognisable (large pitch/roll cause nauseating views).
        self._pitch_range = (-25.0, 25.0)   # wider: typical seated VR viewing
        self._roll_range  = (-18.0, 18.0)   # wider: head tilt during casual browsing

        # Synthetic artifact augmentation (applied per-image before patch extraction).
        # prob=0.3 is the recommended default. For small datasets (< 400 images) use
        # 0.1–0.2; for large combined datasets (thousands of images) 0.3–0.5 is safe.
        # The caller controls this via artifact_aug_prob rather than relying on the
        # class-level default, so that train_on() can pass a dataset-size-aware value.
        self._artifact_aug = SyntheticArtifactAugmentation(prob=artifact_aug_prob)

        self.rotators = {} 

    def get_rotator(self, h, w):
        if (h, w) not in self.rotators:
            # Always keep rotators on CPU — augmentation runs in DataLoader workers
            # which have no CUDA context.  Moving to GPU here would crash with
            # "Cannot re-initialize CUDA in forked subprocess".
            self.rotators[(h, w)] = SphericalRotation(h, w, device='cpu')
        return self.rotators[(h, w)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        score = torch.tensor(float(self.scores[idx])).float()
        
        img_pil = Image.open(img_path).convert('RGB')
        w, h = img_pil.size

        # Stereo flag is set at the dataset level (passed via the ds spec dict).
        # Do not attempt filename or shape heuristics — both are unreliable.
        is_stereo = self.stereo

        # Handle stereoscopic images (Top-Bottom ODS).
        # Crop in PIL *before* converting to a tensor: this avoids loading the full
        # double-height image into RAM and converting unnecessary pixels.
        if is_stereo:
            if self.augment:
                # Randomly pick top or bottom view as a form of augmentation
                if random.random() > 0.5:
                    img_pil = img_pil.crop((0, h // 2, w, h))  # Bottom view
                else:
                    img_pil = img_pil.crop((0, 0, w, h // 2))  # Top view
            else:
                img_pil = img_pil.crop((0, 0, w, h // 2))  # Always top view for eval

            h = h // 2  # Update h to reflect single-view height

        if w > 2048 or h > 1024:
            scale_factor = min(2048 / w, 1024 / h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = new_w, new_h

        if self.augment:
            img_pil = self.jitter(img_pil)

        img_tensor = F.to_tensor(img_pil)  # (C, H, W) on CPU
        del img_pil # Explicitly free PIL image memory to reduce CPU RAM footprint

        if self.augment:
            # ── Spatially-consistent synthetic artifacts ──────────────────────
            # Sample ONE artifact configuration per image and apply to the ERP
            # BEFORE projection, so all 6 cube faces share the same distortion.
            # Per-face independent artifacts would teach unrealistic spatial
            # discontinuity at face boundaries.
            artifact_params = self._artifact_aug.sample_params()

            rotator = self.get_rotator(h, w)  # h is already updated to single-view height
            # ── Fused Rotation + Cubemap Projection ──────────────────────────
            # Instead of:
            #   1. rotator.rotate()               → bilinear resample ERP→ERP
            #   2. equirectangular_to_cube_faces() → bilinear resample ERP→cube
            # we compose the rotation matrix with the cube-face direction grids
            # and perform a SINGLE grid_sample from the original ERP directly
            # into the 6 cube faces.  This eliminates the double-interpolation
            # blur that would otherwise act as a low-pass filter on the input.
            # 70% of the time draw yaw uniformly (full coverage).
            # 30% of the time bias toward the equatorial band where most content lies.
            if random.random() < 0.7:
                yaw = torch.rand(1).item() * 360.0
            else:
                # Gaussian centred on 0° yaw (front), σ=60° — stays in equatorial band
                yaw = (random.gauss(0.0, 60.0)) % 360.0
            pitch = random.uniform(*self._pitch_range)
            roll  = random.uniform(*self._roll_range)
            R = rotator._create_rotation_matrix(roll=roll, pitch=pitch, yaw=yaw)

            # Normalize to [-1, 1] before the fused projection
            img_tensor = (img_tensor - 0.5) * 2.0

            # Apply artifact to ERP (spatially consistent across all faces)
            if artifact_params is not None:
                img_tensor = self._artifact_aug.apply_params(img_tensor, artifact_params)

            # rotate_and_project expects (1, C, H, W)
            cube_faces_tensor = rotate_and_project(img_tensor.unsqueeze(0), R)

        elif (self.yaw is not None and self.yaw != 0) or (getattr(self, 'pitch', None) is not None and self.pitch != 0) or (getattr(self, 'roll', None) is not None and self.roll != 0):
            # TTA path: fused rotation for a TTA pass
            rotator = self.get_rotator(h, w)
            R = rotator._create_rotation_matrix(roll=getattr(self, 'roll', 0) or 0, pitch=getattr(self, 'pitch', 0) or 0, yaw=getattr(self, 'yaw', 0) or 0)
            img_tensor = (img_tensor - 0.5) * 2.0
            cube_faces_tensor = rotate_and_project(img_tensor.unsqueeze(0), R)

        else:
            # No rotation: straight projection, no rotation matrix needed
            img_tensor = (img_tensor - 0.5) * 2.0
            cube_faces_tensor = equirectangular_to_cube_faces(img_tensor, device='cpu')

        # Free the large ERP image tensor immediately after projection
        del img_tensor

        # Viewport-aware face dropout (training only).
        # With probability 0.15, zero out one of the 4 equatorial faces.
        # This forces the model to infer global quality from 5 of 6 faces,
        # improving robustness to partially occluded or corrupted views.
        # Never drop polar faces (indices 2, 3) — they are already underweighted.
        drop_face = -1
        if self.augment and random.random() < 0.15:
            equatorial_face_indices = [0, 1, 4, 5]          # +X, -X, +Z, -Z
            drop_face = random.choice(equatorial_face_indices)

        patch_size = self.hse_opts['patch_size']
        patch_pixels = 3 * patch_size * patch_size  # e.g. 3*32*32 = 3072

        all_patches, all_spatial, all_scale, all_face_ids, all_patch_masks = [], [], [], [], []

        for i in range(cube_faces_tensor.shape[1]):  # iterate over 6 faces
            face_tensor = cube_faces_tensor[0, i]

            if self.augment:
                face_tensor = face_tensor.clamp(-1.0, 1.0)

            # Returns (1, N, patch_pixels + 3) — squeeze batch dim -> (N, 3075)
            flat = get_multiscale_patches(face_tensor, **self.hse_opts).squeeze(0)

            pixel_data = flat[:, :patch_pixels]           # (N, 3072)
            spatial    = flat[:, patch_pixels].long()     # (N,)
            scale      = flat[:, patch_pixels + 1].long() # (N,)
            # flat[:, patch_pixels + 2] is the mask — not needed here

            # Reshape pixels back into patch tensors: (N, C, H, W)
            patches = pixel_data.reshape(-1, 3, patch_size, patch_size)

            n_patches = patches.shape[0]

            if i == drop_face:
                patches = torch.zeros_like(patches)
                face_mask = torch.zeros(n_patches, dtype=torch.float32)
            else:
                face_mask = torch.ones(n_patches, dtype=torch.float32)

            all_patches.append(patches)
            all_spatial.append(spatial)
            all_scale.append(scale)
            all_face_ids.append(torch.full_like(spatial, i))
            all_patch_masks.append(face_mask)

        all_patches_tensor = torch.cat(all_patches, dim=0)

        out_dict = {
            'patches':     all_patches_tensor,  # (6*N, C, H, W)
            'spatial_pos': torch.cat(all_spatial,   dim=0),  # (6*N,)
            'scale_pos':   torch.cat(all_scale,     dim=0),  # (6*N,)
            'face_ids':    torch.cat(all_face_ids,  dim=0),  # (6*N,)
            'patch_masks': torch.cat(all_patch_masks, dim=0),
            'score':       score,
        }
        if self.scene_ids is not None:
            out_dict['scene_id'] = torch.tensor(self.scene_ids[idx])
        return out_dict
