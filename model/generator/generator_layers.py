import torch
from torch import nn

from model.utils import MaskedAttention


class GRU(nn.Module):
    def __init__(self, code_num, hidden_dim, max_len, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.device = device

        self.gru_cell = nn.GRUCell(input_size=code_num, hidden_size=hidden_dim)
        self.hidden2codes = nn.Sequential(
            nn.Linear(hidden_dim, code_num),
            nn.Sigmoid()
        )

    def step(self, x, h=None):
        h_n = self.gru_cell(x, h)
        codes = self.hidden2codes(h_n)
        return codes, h_n

    def forward(self, noise):
        codes = self.hidden2codes(noise)
        h = torch.zeros(len(codes), self.hidden_dim, device=self.device)
        samples, hiddens = [], []
        for _ in range(self.max_len):
            samples.append(codes)
            codes, h = self.step(codes, h)
            hiddens.append(h)
        samples = torch.stack(samples, dim=1)
        hiddens = torch.stack(hiddens, dim=1)

        return samples, hiddens




class SmoothCondition(nn.Module):
    def __init__(self, code_num, attention_dim, sigma=1.0):
        super().__init__()
        self.attention = MaskedAttention(code_num, attention_dim)
        self.sigma = sigma  # Tham số điều chỉnh độ rộng của Gaussian Kernel

    def gaussian_kernel(self, x):
        """
        Tính ma trận Gaussian Kernel cho dữ liệu đầu vào x.
        x có shape (batch_size, seq_len, code_num)
        """
        x_i = x.unsqueeze(2)  # (batch_size, seq_len, 1, code_num)
        x_j = x.unsqueeze(1)  # (batch_size, 1, seq_len, code_num)
        dist = torch.norm(x_i - x_j, dim=-1) ** 2  # Bình phương khoảng cách
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))  # Áp dụng Gaussian
        return kernel  # (batch_size, seq_len, seq_len)

    def forward(self, x, lens, target_codes):
        score = self.attention(x, lens)  # (batch_size, seq_len)
        kernel = self.gaussian_kernel(x)  # (batch_size, seq_len, seq_len)

        # Nhân Kernel với score để tạo ra giá trị điều chỉnh
        weighted_score = torch.bmm(kernel, score.unsqueeze(-1)).squeeze(-1)

        score_tensor = torch.zeros_like(x)
        score_tensor[torch.arange(len(x)), :, target_codes] = weighted_score
        x = x + score_tensor
        x = torch.clip(x, max=1)
        return x

