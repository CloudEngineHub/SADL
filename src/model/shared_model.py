"""One cross-task predictor and actor, with a task-independent padded action API."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from jepa import JEPA


def attend(q, k, v):
    # Structural masks are decomposed into unmasked calls. Flash is used for BF16 training;
    # FP32 evaluation falls back to the efficient / math kernels.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


class CoreBlock(nn.Module):
    def __init__(self, dim, heads, modal):
        super().__init__()
        self.dim, self.heads, self.modal = dim, heads, modal
        count = 2 if modal else 1
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(count)])
        self.qkv = nn.ModuleList([nn.Linear(dim, 3 * dim) for _ in range(count)])
        self.out = nn.ModuleList([nn.Linear(dim, dim) for _ in range(count)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
            nn.GELU(), nn.Linear(4 * dim, dim)) for _ in range(count)])

    def project(self, x, branch):
        parts = self.qkv[branch](self.norm[branch](x)).chunk(3, -1)
        return [p.reshape(*p.shape[:2], self.heads, self.dim // self.heads).transpose(1, 2)
                for p in parts]

    def forward(self, state, action, actor=False):
        ns, na = state.shape[1], action.shape[1]
        qs, ks, vs = self.project(state, 0)
        qa, ka, va = self.project(action, int(self.modal))
        if actor:
            ys = attend(qs, ks, vs)
            ya = attend(qa, torch.cat((ks, ka), 2), torch.cat((vs, va), 2))
        else:
            if ns != na:
                raise ValueError('Prediction needs aligned state/action tokens')
            state_values, action_values = [], []
            # Exactly the old block-causal mask: both modalities at t see <= t.
            for t in range(ns):
                query = torch.cat((qs[:, :, t:t+1], qa[:, :, t:t+1]), 2)
                value = attend(query, torch.cat((ks[:, :, :t+1], ka[:, :, :t+1]), 2),
                               torch.cat((vs[:, :, :t+1], va[:, :, :t+1]), 2))
                state_values.append(value[:, :, :1])
                action_values.append(value[:, :, 1:])
            ys, ya = torch.cat(state_values, 2), torch.cat(action_values, 2)
        state = state + self.out[0](ys.transpose(1, 2).reshape_as(state))
        action = action + self.out[int(self.modal)](ya.transpose(1, 2).reshape_as(action))
        return state + self.ff[0](state), action + self.ff[int(self.modal)](action)


class SharedCore(nn.Module):
    def __init__(self, dim, depth, heads, modal=False):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 8, dim) * .02)
        self.modality = nn.Parameter(torch.randn(2, 1, dim) * .02)
        self.blocks = nn.ModuleList([CoreBlock(dim, heads, modal) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2 if modal else 1)])
        self.modal = modal

    def forward(self, state, action, actor=False):
        state = state + self.position[:, :state.shape[1]] + self.modality[0]
        action = action + self.position[:, :action.shape[1]] + self.modality[1]
        for block in self.blocks:
            state, action = block(state, action, actor)
        return self.norm[0](state), self.norm[int(self.modal)](action)


class SharedPredictor(nn.Module):
    def __init__(self, dim, core):
        super().__init__()
        self.output = nn.Linear(dim, dim)
        object.__setattr__(self, 'core', core)

    @property
    def pos_embedding(self):
        return self.core.position[:, :3]

    def forward(self, z, action):
        state, _ = self.core(z, action, actor=False)
        return self.output(state)


class DenseMoE(nn.Module):
    """All tasks use the same soft gate and experts; there is no task-ID routing."""
    def __init__(self, inputs, hidden, outputs, experts):
        super().__init__()
        self.gate = nn.Linear(inputs, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(inputs, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU()) for _ in range(experts)])
        self.output = nn.Linear(hidden, outputs)

    def forward(self, x):
        weights = self.gate(x).softmax(-1)
        values = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return self.output((values * weights.unsqueeze(-1)).sum(-2))


class SharedActor(nn.Module):
    def __init__(self, dim, action_dim, hidden, experts, moe, core):
        super().__init__()
        self.embed_dim, self.action_dim = dim, action_dim
        self.min_log_std, self.max_log_std = -5.0, 2.0
        self.feature_layout = 'four_slot'
        object.__setattr__(self, 'core', core)
        if moe:
            self.net = DenseMoE(4 * dim, hidden, 2 * action_dim, experts)
        else:
            self.net = nn.Sequential(nn.Linear(4 * dim, hidden), nn.LayerNorm(hidden),
                nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, 2 * action_dim))

    def forward(self, z, intent, previous):
        shape = z.shape
        state = torch.stack((z, z + intent), dim=-2).reshape(-1, 2, shape[-1])
        _, query = self.core(state, previous.reshape(-1, 1, shape[-1]), actor=True)
        features = torch.cat((z, intent, z * intent, query.reshape(shape)), -1)
        mean, log_std = self.net(features).chunk(2, -1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def action_mean(self, z, intent, previous):
        value = self(z, intent, previous)[0]
        return value[..., :getattr(self, 'deployment_action_dim', self.action_dim)]


class PaddedActionEncoder(nn.Module):
    def __init__(self, original, action_dim):
        super().__init__()
        self.base, self.action_dim = original, action_dim

    def forward(self, action):
        if action.shape[-1] > self.action_dim:
            raise ValueError('Action exceeds the shared padded interface')
        padded = F.pad(action, (0, self.action_dim - action.shape[-1]))
        return self.base(padded[:, None])[:, 0] if action.ndim == 2 else self.base(padded)


class SharedJEPA(JEPA):
    def __init__(self, *, downstream, core_depth=2, core_heads=4, actor_hidden=576,
                 actor_experts=4, task_action_dim=0, **kwargs):
        super().__init__(**kwargs)
        dim, action_dim = self.intent_actor.embed_dim, self.intent_actor.action_dim
        if downstream not in ('transformer_mlp', 'transformer_moe', 'mot_mlp'):
            raise ValueError(downstream)
        self.downstream = downstream
        self.task_action_dim = task_action_dim
        self.max_action_dim = action_dim
        self.shared_core = SharedCore(dim, core_depth, core_heads, downstream == 'mot_mlp')
        self.predictor = SharedPredictor(dim, self.shared_core)
        self.intent_actor = SharedActor(dim, action_dim, actor_hidden, actor_experts,
                                       downstream == 'transformer_moe', self.shared_core)
        self.intent_actor.deployment_action_dim = task_action_dim or action_dim
        self.action_encoder = PaddedActionEncoder(self.action_encoder, action_dim)

    def get_action_dim(self, fallback=None):
        return self.task_action_dim or self.max_action_dim

    def action_parameters(self, z, intent, previous_action, previous_embedding=None):
        values = super().action_parameters(z, intent, previous_action, previous_embedding)
        dim = self.get_action_dim()
        return tuple(value[..., :dim] for value in values)
