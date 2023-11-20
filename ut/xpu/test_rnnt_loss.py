import torch


def transducer_loss_reference(x, label, f_len, y_len, blank_idx, loss_grad):
    def log_sum_exp(a, b):
        if (a >= b):
            return a + torch.log(1 + torch.exp(b-a))
        else:
            return b + torch.log(1 + torch.exp(a-b))

    def forward_alpha(x, label, f_len, y_len, blank_idx):
        B, T, U, V = x.size()
        acc_t = torch.float32 if x.dtype in [torch.float16, torch.float32] else x.dtype
        alpha = torch.zeros((B, T, U), dtype=acc_t, device=x.device)
        for b in range(B):
            alpha[b, 0, 0] = 0
            for t in range(1, f_len[b]):
                alpha[b, t, 0] = alpha[b, t-1, 0] + x[b, t-1, 0, blank_idx]
            for u in range(1, y_len[b]+1):
                alpha[b, 0, u] = alpha[b, 0, u-1] + x[b, 0, u-1, label[b, u-1]]
            for t in range(1, f_len[b]):
                for u in range(1, y_len[b]+1):
                    curr_ = alpha[b, t-1, u] + x[b, t-1, u, blank_idx]
                    next_ = alpha[b, t, u-1] + x[b, t, u-1, label[b, u-1]]
                    alpha[b, t, u] = log_sum_exp(curr_, next_)
        return alpha

    def forward_beta(x, label, f_len, y_len, blank_idx):
        B, T, U, V = x.shape
        acc_t = torch.float32 if x.dtype in [torch.float16, torch.float32] else x.dtype
        beta = torch.zeros((B, T, U), dtype=acc_t, device=x.device)
        for b in range(B):
            beta[b, f_len[b]-1, y_len[b]] = x[b, f_len[b]-1, y_len[b], blank_idx]
            for t in range(f_len[b]-2, -1, -1):
                beta[b, t, y_len[b]] = beta[b, t+1, y_len[b]] + x[b, t, y_len[b], blank_idx]
            for u in range(y_len[b]-1, -1, -1):
                beta[b, f_len[b]-1, u] = beta[b, f_len[b]-1, u+1] + x[b, f_len[b]-1, u, label[b, u]]
            for t in range(f_len[b]-2, -1, -1):
                for u in range(y_len[b]-1, -1, -1):
                    curr_ = beta[b, t+1, u] + x[b, t, u, blank_idx]
                    next_ = beta[b, t, u+1] + x[b, t, u, label[b, u]]
                    beta[b, t, u] = log_sum_exp(curr_, next_)
        return beta

    def backward(x, label, f_len, y_len, alpha, beta, loss_grad, blank_idx):
        grad = torch.zeros_like(x)
        B, T, U, V = x.size()
        for b in range(B):
            common_factor = torch.log(loss_grad[b]) + alpha - beta[b, 0, 0]
            # next
            for u in range(y_len[b]):
                grad[b, :f_len[b], u, label[b, u]] = -torch.exp(common_factor[b, :f_len[b], u]
                                                        + beta[b, :f_len[b], u+1]
                                                        + x[b, :f_len[b], u, label[b, u]])

            # current
            grad[b, :f_len[b]-1, :y_len[b]+1, blank_idx] \
                = -torch.exp(common_factor[b, :f_len[b]-1, :y_len[b]+1]
                    + beta[b, 1:f_len[b], :y_len[b]+1]
                    + x[b, :f_len[b]-1, :y_len[b]+1, blank_idx])

            grad[b, f_len[b]-1, y_len[b], blank_idx] = -torch.exp(common_factor[b, f_len[b]-1, y_len[b]]
                                                         + x[b, f_len[b]-1, y_len[b], blank_idx])

        return grad

    x_log = torch.nn.functional.log_softmax(x, dim=-1)
    alpha = forward_alpha(x_log, label, f_len, y_len, blank_idx)
    beta = forward_beta(x_log, label, f_len, y_len, blank_idx)
    grad = backward(x_log, label, f_len, y_len, alpha, beta,
                        loss_grad, blank_idx)
    x_log.backward(grad)
    loss = -beta[:, 0, 0]
    loss = loss.to(x.dtype)
    return alpha, beta, x.grad, loss

for_vector_kernel=False
scalar_t = torch.float
B = 5
T_min = 23
T_max = 51
U_min = 12
U_max = 25
V = 16 if for_vector_kernel else 14
blank_idx = V - 1
device = "cpu"

x_tst = torch.randn((B, T_max, U_max, V), dtype=scalar_t, requires_grad=True, device=device)  # [B, T, U, V]
y = torch.randint(0, blank_idx, (B, U_max-1), dtype=torch.int, device=device)  # [B, U_max-1]
f_len = torch.randint(T_min, T_max+1, (B,), dtype=torch.int, device=device)    # [B]
y_len = torch.randint(U_min-1, U_max, (B,), dtype=torch.int, device=device)    # [B]
f_len[torch.randint(0, B, (1,)).item()] = T_max
y_len[torch.randint(0, B, (1,)).item()] = U_max-1
print("y={}".format(y))
print("f_len={}".format(f_len))
print("y_len={}".format(y_len))

# x_tst_packed, self.batch_offset = _pack(x_tst)
# Generate reference
x_ref = x_tst.data.clone()
x_ref.requires_grad = True
loss_grad = torch.ones(x_ref.size(0), dtype=x_ref.dtype, device=x_ref.device)/x_ref.size(0)
_, _, grad_ref, loss_ref = transducer_loss_reference( x=x_ref,
                                                    label=y,
                                                    f_len=f_len,
                                                    y_len=y_len,
                                                    blank_idx=blank_idx,
                                                    loss_grad=loss_grad)
print("loss={}".format(loss_ref))
