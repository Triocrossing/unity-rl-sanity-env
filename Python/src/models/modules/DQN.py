import torch
from torch import nn


class DQN(nn.Module):
    """Simple Deep Q-Network.

    :param n_actions: number of discrete actions available in the env.
    """

    def __init__(self, n_actions: int):
        super().__init__()

        self.obstacle_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.camera_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, n_actions)
        )

    def forward(
        self, obstacle_state: torch.Tensor, camera_state: torch.Tensor
    ) -> torch.Tensor:
        obstacle_out = self.obstacle_fc(obstacle_state)
        camera_out   = self.camera_fc(camera_state)
        out = self.fuse_fc(torch.cat([camera_out, obstacle_out], axis=1))
        return out

class DQN_occupancy_grid(nn.Module):
  """
  Deep Q-Network with occupancy grid

  :param n_actions: number of discrete actions available in the env.
  """

  def __init__(self, n_actions: int):
    super().__init__()

    self.env_encoder = nn.Sequential(
      nn.Conv3d(in_channels=1, out_channels=8,  kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
    )
    
    self.env_fc = nn.Sequential(
      nn.Linear(432, 64),
      nn.ReLU()
    )

    self.position_fc = nn.Sequential(
      nn.Linear(3, 64),
      nn.ReLU(),
    )

    self.C_value = nn.Sequential(
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, n_actions)
    )
  
  def forward(self, obstacle_state: torch.Tensor, camera_state: torch.Tensor) -> torch.Tensor:
    # print("obstacle_state,", obstacle_state.size())
    batch_size = obstacle_state.size(0)
    self.env_encoder.cuda()
    obstacle_state = obstacle_state.cuda()
    # obstacle_state = obstacle_state.cuda()
    # env_encoder = self.env_encoder.cuda()

    obstacle_state_encode = self.env_encoder(obstacle_state).view(batch_size, -1)
    # print("obstacle_state_encode,", obstacle_state_encode.size())
    obstacle_out = self.env_fc(obstacle_state_encode)
    camera_out = self.position_fc(camera_state)

    return self.C_value(torch.cat([obstacle_out, camera_out], axis=1))