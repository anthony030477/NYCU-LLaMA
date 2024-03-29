import torch
import torch.nn.functional as F

def cos_sim(a:torch.Tensor, b:torch.Tensor):
    return (a @ b.T)/(torch.norm(a,dim=1)[:,None]@torch.norm(b,dim=1)[None,:])


def infonNCE_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    temperature: float = 0.5,
):
    '''
    u:(N,d), v:(M,d), out:(N,M)
    '''
    sim=cos_sim(u,v)/temperature
    return -torch.diagonal(torch.log_softmax(sim, dim=1)).mean()


def simCLR_loss(
    u: torch.Tensor,                               # [N, C]
    v: torch.Tensor,                               # [N, C]
    temperature: float = 0.5,
):
    """
    N: batch size
    C: feature dimension
    """
    N, C = u.shape

    z = torch.cat([u, v], dim=0)                   # [2N, C]
    z = F.normalize(z, p=2, dim=1)                 # [2N, C]
    s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
    s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity
    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N),                           # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss
    return loss
