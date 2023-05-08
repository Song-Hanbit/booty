import torch
import numpy as np
print('device:',
      (device := torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
torch.set_grad_enabled(False)


class _Gaussian(torch.distributions.normal.Normal):
    def __init__(self): super().__init__(0, 1)


# TODO: CUDA OOM 대비 buffer 기능 추가; dim 제외한 나머지 차원에 대해 분할
class _Resampling:
    def __init__(self, data, statistics, dim=0, 
                 dim_kw=None, ensured_torch=False, **kwarg):
        data = self.__init_start(data, statistics, dim)
        resampled_sample = self._resampling_method(data, **kwarg)
        self.__init_get_resampled_stat(resampled_sample, dim_kw, ensured_torch)

    def __init_start(self, data, statistics, dim):
        self.data = torch.as_tensor(data, device=device)
        self.dim = _rectify_dim(data, dim)
        self.statistics = statistics
        return data.permute(self.dim, 
                            *[i for i in range(len(data.shape)) if i != self.dim])

    def _resampling_method(self, data, **kwarg):
        raise NotImplementedError(
            '_Resampling should be inherited by a class with defined method.')

    def __init_get_resampled_stat(self, resampled_sample, dim_kw, ensured_torch):
        has_no_dim = False
        # not torch and not torch derivatives
        if self.statistics.__module__ != 'torch' and not ensured_torch:
            if dim_kw is None:
                if self.statistics.__module__ == 'numpy': 
                    self.dim_kw = 'axis'
                else: 
                    has_no_dim = True
            else:
                self.dim_kw = dim_kw
            resampled_sample = np.asarray(resampled_sample.cpu())
        # torch or torch derivatives
        elif dim_kw is None and (self.statistics.__module__ == 'torch' 
                                 or ensured_torch):
            self.dim_kw = 'dim'
        else: 
            self.dim_kw = dim_kw
        if has_no_dim: 
            stat = self.statistics(resampled_sample)
            theta_hat = self.statistics(self.data)
        else: 
            stat = self.statistics(resampled_sample, **{self.dim_kw:0})
            theta_hat = self.statistics(self.data, **{self.dim_kw:self.dim})
        resampled = torch.as_tensor(stat, device=device)
        dim_order = list(range(len(self.data.shape)))
        dim_order.insert(self.dim, dim_order.pop(0))
        self.resampled = resampled.permute(dim_order)
        self._theta_hat = theta_hat.unsqueeze(self.dim)   

    def get_error(self): return self.resampled.std(dim=self.dim, keepdim=True)

    def get_quantile_ci(self, confidence=0.95):
        half_alpha = (1 - confidence) / 2
        q = torch.tensor([half_alpha, 1 - half_alpha], device=device)
        return torch.quantile(self.resampled, q, dim=self.dim, keepdim=True
                              ).transpose(0,
                                          self.dim - len(self.data.shape)
                                          ).squeeze()

    def get_reverse_quantile_ci(self, confidence=0.95):
        return torch.flip(2 * self._theta_hat - self.get_quantile_ci(confidence),
                          (self.dim,))


class Jackknife(_Resampling):
    def __init__(self, data, statistics, dim=0, remove=1, dim_kw=None, 
                 ensured_torch=False):
        super().__init__(data, statistics, dim=dim, dim_kw=dim_kw, 
                         ensured_torch=ensured_torch, remove=remove)

    def _resampling_method(self, data, remove):
        num_sample = data.shape[self.dim]
        idxs = torch.arange(num_sample)
        comb = torch.combinations(idxs, remove)
        sample_idxs = idxs.unsqueeze(0) * torch.ones_like(comb)
        sample_idxs = sample_idxs[sample_idxs != comb].view(-1, num_sample - remove).T
        return data[sample_idxs]


class Bootstrap(_Resampling):
    def __init__(self, data, statistics, dim=0, resample_size=None, 
                 resample_times=10000, dim_kw=None, ensured_torch=False):
        super().__init__(data, statistics, dim=dim, dim_kw=dim_kw, 
                         ensured_torch=ensured_torch, 
                         resample_size=resample_size, resample_times=resample_times)

    def _resampling_method(self, data, resample_size, resample_times):
        if resample_size is None: resample_size = data.shape[self.dim]
        sample_idxs = torch.randint(data.shape[0], 
                                    (resample_size, resample_times))
        return data[sample_idxs] 

    def get_bca_ci(self, confidence=0.95):
        jk = Jackknife(self.data, self.statistics, self.dim, dim_kw=self.dim_kw)
        jk_bias = jk.resampled.mean(dim=self.dim, keepdim=True) - jk.resampled
        acceleration = (jk_bias ** 3).sum(dim=self.dim, keepdim=True) \
                     / 6 \
                     / ((jk_bias ** 2).sum(dim=self.dim, keepdim=True)) ** (3 / 2)
        gaussian = _Gaussian()
        half_alpha = torch.tensor((1 - confidence) / 2, device=device)
        quantile = discrete_cdf(self.resampled, self._theta_hat, dim=self.dim)        
        z_0_hat = gaussian.icdf(quantile)
        z_half_alpha = gaussian.icdf(half_alpha)
        z_q_pos = z_0_hat + z_half_alpha
        z_q_neg = z_0_hat - z_half_alpha
        z = torch.cat([z_0_hat + z_q_pos / (1 - acceleration * z_q_pos),
                       z_0_hat + z_q_neg / (1 - acceleration * z_q_neg)], 
                       dim=self.dim)
        return discrete_quantile(self.resampled, gaussian.cdf(z), dim=self.dim)


def is_integer(x):
    ints = [int, torch.int8, torch.int16, torch.int32, torch.int64,
            np.int8, np.int16, np.int32, np.int64]
    return True if type(x) in ints else False 

def _rectify_dim(data, dim): 
    if not is_integer(dim): 
        raise RuntimeError('dim should be integer of Python, NumPy or Torch')
    if dim < -(num_dims := len(data.shape)) or dim > num_dims - 1:
        raise RuntimeError(f'dim should be in [-{num_dims}, {num_dims - 1}]')
    return dim if dim >= 0 else num_dims + dim

# TODO: interpolation 부분 다시
# TODO: input validation
def discrete_cdf(samples, quantile, dim=0, interpolation='midpoint'):
    while (len(samples.shape) < len(quantile.shape) or 
            len(samples.shape) <= len(quantile.shape) and quantile.shape[dim] > 1): 
        samples = samples.unsqueeze(-1)
    if interpolation == 'midpoint':
        q = (((samples < quantile).sum(dim=dim) + (samples <= quantile).sum(dim=dim)
            ) / (2 * samples.shape[dim])).unsqueeze(dim)
    elif interpolation == 'higher':
        q = ((samples <= quantile).sum(dim=dim) / samples.shape[dim]).unsqueeze(dim)
    elif interpolation == 'lower':
        q = ((samples < quantile).sum(dim=dim) / samples.shape[dim]).unsqueeze(dim)        
    return q

# TODO: interpolation 구현 필요
# TODO: input validation
def discrete_quantile(samples, q, dim=0):
    dim = _rectify_dim(samples, dim)
    samples = samples.sort(dim=dim)[0]
    num_samples = samples.shape[dim]
    num_dims = len(samples.shape)
    idxs = (q * (num_samples - 1)).round().type(torch.long)
    broadcast = list()
    for _dim in range(num_dims):
        if dim == _dim: broadcast.append(idxs)
        else: 
            view_arg = torch.ones(num_dims).type(torch.long)
            view_arg[_dim] = -1
            broadcast.append(torch.arange(idxs.shape[_dim]).view(view_arg.tolist()))
    return samples[broadcast]