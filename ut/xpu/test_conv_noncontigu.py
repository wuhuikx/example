14:02:15      def test_conv_noncontig_weights(self, device):
14:02:15          for dim in (1, 2, 3):
14:02:15              for grouped in (False, True):
14:02:15                  nc = 3
14:02:15                  groups = 3 if grouped else 1
14:02:15                  w = torch.randn([3] * dim, device=device)
14:02:15                  w = w.expand([nc, int(nc / groups)] + list(w.shape))
14:02:15                  w = w.detach().requires_grad_()
14:02:15                  x = torch.randn([1, nc] + ([5] * dim), device=device, requires_grad=True)
14:02:15                  y = getattr(F, 'conv{}d'.format(dim))(x, w, groups=groups)
14:02:15                  y.sum().backward()
14:02:15  >               y = getattr(F, 'conv_transpose{}d'.format(dim))(x, w, groups=groups)
