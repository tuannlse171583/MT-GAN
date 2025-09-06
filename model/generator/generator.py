import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseModel
from model.utils import sequence_mask
from .generator_layers import SmoothCondition  # giữ nguyên để bơm điều kiện


# ---------- Code-level head: Masked MLP (autoregressive theo chiều mã) ----------
class _AutoregressiveLinear(nn.Linear):
    """
    Linear với mask tam giác dưới để đảm bảo phụ thuộc tự hồi quy giữa các mã trong cùng 1 visit.
    """
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        # mask: chỉ cho phép phụ thuộc vào chính nó và các mã "trước" nó
        mask = torch.tril(torch.ones(out_features, in_features))
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: (B, V) -> (B, V) nếu in_features == out_features
        W = self.weight * self.mask
        return F.linear(x, W, self.bias)


class _MaskedMLP(nn.Module):
    """
    Hai lớp tuyến tính tự hồi quy (code-level).
    Dùng attention_dim làm hidden width để tương thích tham số repo gốc.
    """
    def __init__(self, vocab_size: int, hidden: int = 1024):
        super().__init__()
        self.fc1 = _AutoregressiveLinear(vocab_size, hidden)
        self.fc2 = _AutoregressiveLinear(hidden, vocab_size)

    def forward(self, x):
        # x: (B, V) multi-hot/prob/sampled của visit hiện tại
        h = F.relu(self.fc1(x))
        return self.fc2(h)  # logits (B, V)


# ---------- Visit-level: HALO-lite Transformer ----------
class _VisitPositional(nn.Module):
    def __init__(self, n_ctx: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(n_ctx, d_model)

    def forward(self, x, t_idx):
        # x: (B, T, D), t_idx: (B, T)
        return x + self.pe(t_idx)


class _HaloLiteVisitModel(nn.Module):
    """
    Transformer decoder nông (causal) để sinh chuỗi hidden theo thời gian visit.
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int, n_ctx: int, ff_dim: int = 4*256, dropout: float = 0.1):
        super().__init__()
        self.n_ctx = n_ctx
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.pos = _VisitPositional(n_ctx, d_model)

        # memory 1 token học được (giống "global context")
        self.mem_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mem_token, std=0.02)

    def forward(self, h_in):
        """
        h_in: (B, T, D) – input query cho decoder (đã trộn noise/condition)
        return: H (B, T, D)
        """
        B, T, D = h_in.shape
        device = h_in.device

        # add position
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h_in = self.pos(h_in, t_idx)

        # causal mask T x T (True ở phía trên đường chéo -> bị mask)
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        # memory: bản sao mem_token theo batch
        mem = self.mem_token.expand(B, -1, -1)  # (B,1,D)

        H = self.dec(tgt=h_in, memory=mem, tgt_mask=causal)
        return H  # (B,T,D)


# ---------- Generator thay thế GRU bằng HALO-lite ----------
class Generator(BaseModel):
    """
    Drop-in thay thế cho GRU Generator gốc:
    - Visit-level: Transformer decoder (causal)
    - Code-level: Masked MLP autoregressive
    - Điều kiện: dùng lại SmoothCondition như repo gốc để tương thích hành vi
    """
    def __init__(self, code_num, hidden_dim, attention_dim, max_len, device=None):
        """
        code_num      : số lượng mã (vocab size theo visit)
        hidden_dim    : d_model cho Transformer (thay cho hidden GRU)
        attention_dim : width cho masked MLP (code-level head)
        max_len       : số visit tối đa
        """
        super().__init__(param_file_name='generator.pt')
        self.code_num = code_num
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.max_len = max_len
        self.device = device

        # noise dùng để khởi tạo chuỗi query ở tầng visit
        self.noise_dim = hidden_dim
        self.noise_proj = nn.Linear(self.noise_dim, self.hidden_dim)

        # "query tokens" cho T visit (học được), được trộn với noise mỗi batch
        self.time_query = nn.Parameter(torch.zeros(1, self.max_len, self.hidden_dim))
        nn.init.normal_(self.time_query, std=0.02)

        # visit-level transformer (nông)
        self.visit_model = _HaloLiteVisitModel(
            d_model=self.hidden_dim,
            n_heads=4,
            n_layers=4,
            n_ctx=self.max_len,
            ff_dim=4*self.hidden_dim,
            dropout=0.1,
        )

        # code-level masked MLP
        self.code_head = _MaskedMLP(vocab_size=self.code_num, hidden=self.attention_dim)

        # giữ SmoothCondition để tương thích hành vi "điều kiện" của MTGAN gốc
        self.smooth_condition = SmoothCondition(self.code_num, self.attention_dim)

    # ---- core forward ----
    def forward(self, target_codes, lens, noise):
        """
        target_codes : (B,) hoặc (B,*) – id/nhãn mục tiêu (đầu vào cho conditioner)
        lens         : (B,) – độ dài hữu hiệu (số visit)
        noise        : (B, noise_dim)
        return:
            prob    : (B, T, V) – xác suất sau khi qua SmoothCondition
            hiddens : (B, T, D) – hidden theo visit (cho critic/ghi nhận)
        """
        B = noise.size(0)
        device = noise.device

        # 1) Visit-level: tạo chuỗi hidden theo thời gian
        #    query = query_token(T,D) + noise(B,1,D)
        z = self.noise_proj(noise).unsqueeze(1)                             # (B,1,D)
        q = self.time_query.expand(B, self.max_len, self.hidden_dim) + z    # (B,T,D)
        H = self.visit_model(q)                                             # (B,T,D)

        # 2) Code-level: với mỗi visit t, sinh logits bằng masked-MLP
        #    Dùng "free-run" giống MTGAN (không teacher-forcing): bắt đầu visit từ vector 0.
        logits_list, prob_list = [], []
        prev_vis = torch.zeros(B, self.code_num, device=device)             # (B,V)

        for t in range(self.max_len):
            logits_t = self.code_head(prev_vis)                             # (B,V)
            prob_t = torch.sigmoid(logits_t)                                # (B,V)

            # Tạo "mẫu" nhị phân để nuôi autoregressive trong cùng visit
            # (nhưng không đẩy ra ngoài; chỉ dùng nội bộ để tính bước sau)
            sample_t = torch.bernoulli(prob_t).to(prob_t.dtype)             # (B,V)
            prev_vis = sample_t  # autoregressive trong 1 visit

            logits_list.append(logits_t)
            prob_list.append(prob_t)

        logits = torch.stack(logits_list, dim=1)                            # (B,T,V)
        prob   = torch.stack(prob_list,  dim=1)                             # (B,T,V)

        # 3) Điều kiện & mask độ dài (giống bản gốc)
        prob = self.smooth_condition(prob, lens, target_codes)              # (B,T,V)

        return prob, H

    # ---- sampling API giữ nguyên hành vi ----
    def sample(self, target_codes, lens, noise=None, return_hiddens=False):
        if noise is None:
            noise = self.get_noise(len(lens))
        with torch.no_grad():
            mask = sequence_mask(lens, self.max_len).unsqueeze(dim=-1)      # (B,T,1)
            prob, hiddens = self.forward(target_codes, lens, noise)         # prob:(B,T,V)
            samples = torch.bernoulli(prob).to(prob.dtype)                   # (B,T,V)
            samples *= mask
            if return_hiddens:
                hiddens = hiddens * mask                                     # broadcast (B,T,D)
                return samples, hiddens
            else:
                return samples

    # ---- helpers giữ nguyên để tương thích ----
    def get_noise(self, batch_size):
        noise = torch.randn(batch_size, self.noise_dim).to(self.device)
        return noise

    def get_target_codes(self, batch_size):
        # giữ nguyên hành vi cũ (random 1 mã mục tiêu); tùy pipeline bạn có thể ghi đè ở ngoài
        codes = torch.randint(low=0, high=self.code_num, size=(batch_size, )).to(self.device)
        return codes
