import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils    import get_schedule_fn

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

writer = SummaryWriter("runs/nca")
window_size = 1024
n = 32
N = n

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_sizes[1], act_dim)
        self.value_head  = nn.Linear(hidden_sizes[1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs: torch.Tensor, determined=False):
        logits, value = self(obs)
        distr = torch.distributions.Categorical(logits=logits)
        if determined:
            action = distr.mode
        else:
            action = distr.sample()
        logp   = distr.log_prob(action)
        
        return action, logp, value, distr.entropy()

def get_patches(state):
    N = state.shape[0]
    grid = state.view(1, 1, N, N) 
    padded_grid = F.pad(grid, (1, 1, 1, 1), mode='constant', value=0)
    patches = padded_grid.unfold(2, 3, 1).unfold(3, 3, 1)
    patches = patches.squeeze(0).squeeze(0)

    return patches

def reward_function(new_tensor):
    with torch.no_grad():
        A = new_tensor.cpu().float() 

    kernels = torch.tensor([
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],  # top

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],  # bottom

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],  # left

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]]   # right
    ], dtype=torch.float32).unsqueeze(1)

    neighbors = F.conv2d(A.unsqueeze(0).unsqueeze(0), kernels, padding=1).squeeze(0) 
    center = A.unsqueeze(0).float()       

    different = (neighbors != center).all(dim=0)

    return different.float().flatten()
    
lr_sched  = get_schedule_fn(1e-4)
n_envs = N*N
n_steps = 128
batch_sz = 1024
gamma, gae_lambda, clip_eps = 0.99, 0.95, 0.05
vf_coef, ent_coef = 0.5, 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.randint(0, 2, (N, N)).to(device).float()
tensor_d = tensor.clone()
policy = MLPActorCritic(9, 2).to(device)

optim = Adam(policy.parameters(), lr=lr_sched(1.0))
buffer = RolloutBuffer(
    buffer_size = n_steps,
    observation_space = gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32),
    action_space = gym.spaces.Discrete(2),
    device = device,
    gae_lambda = gae_lambda,
    gamma = gamma,
    n_envs = n_envs,
)
episode_start = np.ones(n_envs, dtype=bool)
counter = 0
timestamp = 0
def update_tensor():
    global tensor, policy, N, buffer, counter, n_steps, optim, episode_start, timestamp, tensor_d
    with torch.no_grad():
        patches = get_patches(tensor).reshape(N * N, 3*3)
        actions, logp, values, _ = policy.act(patches)
        actions_d, _, _, _ = policy.act(patches, False)
        new_tensor = actions.reshape(N, N).float()
        tensor_d = actions_d.reshape(N, N).float()
        rewards = reward_function(new_tensor)
        rewards_d = reward_function(tensor_d)
        writer.add_scalar("Reward", rewards_d.cpu().numpy().mean(), timestamp)
    timestamp += 1

    
    buffer.add(
        patches.cpu(),
        actions.cpu(),
        rewards.cpu(),                  
        episode_start,
        values.cpu(),
        logp.cpu(),
    )
    episode_start = np.zeros(n_envs, dtype=bool)
    tensor = new_tensor.clone()
    
    counter += 1
    if counter == n_steps:
        counter = 0
        with torch.no_grad():
            patches = get_patches(tensor).reshape(N * N, 3*3)
            last_values = policy(patches)[1]
            buffer.compute_returns_and_advantage(last_values.detach(), episode_start)

        for _ in range(4):
            for batch in buffer.get(batch_sz):
                obs_b, act_b, old_val_b, old_logp_b, adv_b, ret_b = batch
                logits, values = policy(obs_b)
                dist   = torch.distributions.Categorical(logits=logits)
                logp_b = dist.log_prob(act_b)
                entropy_b = dist.entropy()
        
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
    optim.param_groups[0]["lr"] = lr_sched(timestamp / 100000)

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    cell_size = window_size // n
    for y in range(n):
        for x in range(n):
            value = tensor_d[y, x]
            glColor3f(1.0, 1.0, 1.0) if value == 1 else glColor3f(0.0, 0.0, 0.0)

            x0 = x * cell_size
            y0 = y * cell_size

            glBegin(GL_QUADS)
            glVertex2f(x0, y0)
            glVertex2f(x0 + cell_size, y0)
            glVertex2f(x0 + cell_size, y0 + cell_size)
            glVertex2f(x0, y0 + cell_size)
            glEnd()

    glutSwapBuffers()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_size, window_size, 0)
    glMatrixMode(GL_MODELVIEW)

def timer(value):
    update_tensor()
    glutPostRedisplay()
    glutTimerFunc(100, timer, 0)

def mouse_click(button, state, x, y):
    global tensor_d
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        cell_size = window_size // n
        grid_x = x // cell_size
        grid_y = y // cell_size

        if 0 <= grid_x < n and 0 <= grid_y < n:
            tensor_d[grid_y, grid_x] = 1
            glutPostRedisplay()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(window_size, window_size)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"NCA")

    glClearColor(0.0, 0.0, 0.0, 0.0)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMouseFunc(mouse_click)
    glutTimerFunc(0, timer, 0)

    glutMainLoop()

if __name__ == "__main__":
    main()
