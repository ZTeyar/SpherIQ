import torch
import math

from spheriq.utils import display_text

class SphericalRotation:
    def __init__(self, height: int, width: int, device: str = 'cuda'):
        self.height, self.width, self.device = height, width, device
        self.cartesian_coords = None  # Precomputing is expensive and unused by rotate_and_project

    def _precompute_grids(self):
        lon = torch.linspace(0, 2 * math.pi, self.width, device=self.device)
        lat = torch.linspace(-math.pi / 2, math.pi / 2, self.height, device=self.device)
        lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing='xy')
        x = torch.cos(lat_grid) * torch.sin(lon_grid)
        y = torch.sin(lat_grid)
        z = torch.cos(lat_grid) * torch.cos(lon_grid)
        return torch.stack([x, y, z], dim=-1)

    def _create_rotation_matrix(self, roll: float, pitch: float, yaw: float):
        roll, pitch, yaw = map(math.radians, [roll, pitch, yaw])
        
        Ry = torch.tensor([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ], device=self.device, dtype=torch.float32)

        Rz = torch.tensor([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll), math.cos(roll), 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ], device=self.device, dtype=torch.float32)

        return Ry @ Rx @ Rz