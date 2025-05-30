import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import RolloutBuffer

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

writer = SummaryWriter("runs/nca")
window_size = 1024
n = 128
N = n
K = 7

class ConvBetaActorCritic(nn.Module):
    def __init__(self, K, act_dim=4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(act_dim, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, act_dim, K, K)
            conv_out_dim = self.cnn(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(conv_out_dim, 256), nn.ReLU(),
        )

        self.alpha_head = nn.Linear(256, act_dim)
        self.beta_head  = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    # ---------------------------------------------------------------
    def forward(self, obs):                 # obs: (B, 4, K, K)
        x = self.shared(self.cnn(obs))
        alpha = F.softplus(self.alpha_head(x)) + 1.0   # keep >1
        beta  = F.softplus(self.beta_head(x))  + 1.0
        value = self.value_head(x).squeeze(-1)
        return alpha, beta, value

    # one-stop helper for rollout collection
    def act(self, obs):
        alpha, beta, value = self(obs)
        dist   = Beta(alpha, beta)
        action = dist.rsample()                 # (B, 4)
        logp   = dist.log_prob(action).sum(-1)  # sum over 4 dims
        entropy = dist.entropy().sum(-1)
        return action, logp, value, entropy

def get_patches(x, K):
    N = x.shape[-1]
    padding = K // 2
    x_padded = F.pad(x.unsqueeze(0), pad=(padding, padding, padding, padding), mode='circular')
    patches = F.unfold(x_padded, kernel_size=K)
    B = N * N
    patches = patches.squeeze(0).transpose(0, 1).reshape(B, 4, K, K)

    return patches

def reward_function(img, img_diff):
    with torch.no_grad():
        _, N, _ = img.shape
        new_img = torch.clamp(img + img_diff, 0.0, 1.0)
        #total_rewards = img.sum(dim=0).flatten()
        #total_rewards_new = new_img.sum(dim=0).flatten()
        total_rewards = symmetry_reward(img).flatten()
        total_rewards_new = symmetry_reward(new_img).flatten()
        done = (total_rewards_new <= 0.01)

    return total_rewards_new, total_rewards_new - total_rewards, done

def symmetry_reward(img: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, N, _ = img.shape
        flipped_h = torch.flip(img, dims=[2])
        flipped_v = torch.flip(img, dims=[1])

        reward_h = -abs(img - flipped_h).sum(dim=0)
        reward_v = -abs(img - flipped_v).sum(dim=0)

    return (reward_h + reward_v) / 2

def contrast_reward(img: torch.Tensor) -> torch.Tensor:
    global_avg = img.mean(dim=(1, 2), keepdim=True)  # (4, 1, 1)
    deviation = (img - global_avg)  # (4, N, N)

    # Sum across channels (or use mean)
    reward = deviation.sum(dim=0)  # (N, N)

    return reward

def dominant_reward(img: torch.Tensor) -> torch.Tensor:
    R, G, B = img[0], img[1], img[2]
    
    # 1. Find global dominant channel
    total_R = R.sum()
    total_G = G.sum()
    total_B = B.sum()
    
    global_max_idx = torch.argmax(torch.tensor([total_R, total_G, total_B]))

    # 2. Find per-pixel dominant channel
    stacked = torch.stack([R, G, B], dim=0)  # (3, N, N)
    pixel_max_idx = stacked.argmax(dim=0)    # (N, N) values in {0, 1, 2}

    # 3. Reward where pixel max matches global max
    reward = (pixel_max_idx == global_max_idx).float()

    return reward

lr = 1e-3
n_envs = N*N
n_steps = 128
batch_sz = 1024
gamma, gae_lambda, clip_eps = 1, 0.95, 0.2
vf_coef, ent_coef = 0.5, 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.rand(4, N, N).to(device).float()
policy = ConvBetaActorCritic(K).to(device)

optim = Adam(policy.parameters(), lr=lr)
buffer = RolloutBuffer(
    buffer_size = n_steps,
    observation_space = gym.spaces.Box(low=0, high=1, shape=(4, K, K), dtype=np.float32),
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
    device = device,
    gae_lambda = gae_lambda,
    gamma = gamma,
    n_envs = n_envs,
)
episode_start = np.ones(n_envs, dtype=bool)
counter = 0
timestamp = 0
total_timestamp = 100000

texture_id = None

def update_tensor():
    global tensor, episode_start, timestamp, counter
    with torch.no_grad():
        patches = get_patches(tensor, K)
        actions, logp, values, _ = policy.act(patches)        
        y = actions.view(N, N, 4).permute(2, 0, 1)
        y = 2*y - 1
        total_rewards, reward_diff, done = reward_function(tensor, y)
        tensor = tensor + y
        tensor = torch.clamp(tensor, 0.0, 1.0)

    writer.add_scalar("Total Reward", total_rewards.cpu().numpy().mean(), timestamp)

    episode_start = done.cpu().numpy()
    buffer.add(
        patches.cpu(),
        actions.cpu(),
        reward_diff.cpu(),                  
        episode_start,
        values.cpu(),
        logp.cpu(),
    )
    #episode_start = np.zeros(n_envs, dtype=bool)
    #episode_start = done.cpu().numpy()

    timestamp += 1
    counter += 1
    if counter == n_steps:
        counter = 0
        with torch.no_grad():
            patches = get_patches(tensor, K)
            last_values = policy(patches)[-1]
            buffer.compute_returns_and_advantage(last_values.detach(), episode_start)

        for _ in range(4):
            for batch in buffer.get(batch_sz):
                obs_b, act_b, old_val_b, old_logp_b, adv_b, ret_b = batch
                alphas, betas, values = policy(obs_b)
                dist   = Beta(alphas, betas)
                logp_b = dist.log_prob(act_b).sum(-1)
                entropy_b = dist.entropy().sum(-1)
        
                ratio = (logp_b - old_logp_b).exp()
                pg_loss = -torch.min(ratio * adv_b,
                                     torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b).mean()
                value_loss = (ret_b - values).pow(2).mean()
                entropy_loss = -entropy_b.mean()
        
                loss = pg_loss + vf_coef * value_loss + ent_coef * entropy_loss
                optim.zero_grad()
                loss.backward()
                optim.step()

        buffer.reset()
        #episode_start = np.ones(n_envs, dtype=bool)

    optim.param_groups[0]["lr"] = lr * (1 - (timestamp / total_timestamp))

def tensor_to_texture():
    # Convert (4, N, N) -> (N, N, 4) and scale to 0-255
    image_tensor = tensor.cpu().numpy()
    data = np.transpose(image_tensor, (1, 2, 0)) * 255
    data = data.astype(np.uint8)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, n, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    tensor_to_texture()

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(window_size, 0)
    glTexCoord2f(1, 1); glVertex2f(window_size, window_size)
    glTexCoord2f(0, 1); glVertex2f(0, window_size)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_size, 0, window_size)  # bottom-left origin
    glMatrixMode(GL_MODELVIEW)

def timer(value):
    update_tensor()
    glutPostRedisplay()
    glutTimerFunc(100, timer, 0)

def mouse_click(button, state, x, y):
    global tensor
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        cell_size = window_size // n
        grid_x = x // cell_size
        grid_y = (window_size - y) // cell_size  # Flip y-axis

        if 0 <= grid_x < n and 0 <= grid_y < n:
            tensor[:, grid_y, grid_x] = 0  # Set RGBA to 0
            #print(f"Set tensor[:, {grid_y}, {grid_x}] = 0")
            glutPostRedisplay()

def init_texture():
    global texture_id
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(window_size, window_size)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"RGBA Tensor Renderer")

    glClearColor(0.0, 0.0, 0.0, 1.0)
    init_texture()

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse_click)
    glutTimerFunc(0, timer, 0)

    glutMainLoop()

if __name__ == "__main__":
    main()