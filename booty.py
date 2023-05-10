# TODO: docstring 추가
# TODO: device 정의 제대로

import torch
import numpy as np
import itertools
from typing import Union
print('device:',
      (device := torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
torch.set_grad_enabled(False)


class _Gaussian(torch.distributions.normal.Normal):
    def __init__(self): super().__init__(0, 1)


class _Resampling:
    def __init__(self, data:torch.Tensor, statistics:callable, dim:int=0, 
                 dim_kw:Union[str,None]=None, ensured_torch:bool=False, **kwarg):
        self.data = torch.as_tensor(data, device=device)
        self.dim = _rectify_dim(data, dim)
        self.statistics = statistics
        self.ensured_torch = ensured_torch
        self.module = self.statistics.__module__
        self._has_no_dim = False
        data = data.permute(self.dim,
                            *[i for i in range(len(data.shape)) if i != self.dim])
        # not torch and not torch derivatives
        if self.module != 'torch' and not self.ensured_torch:
            if dim_kw is None:
                if self.module == 'numpy': self.dim_kw = 'axis'
                else: self._has_no_dim = True
            else:
                self.dim_kw = dim_kw
        # torch or torch derivatives
        elif dim_kw is None and (self.module == 'torch' or self.ensured_torch):
            self.dim_kw = 'dim'
        else: 
            self.dim_kw = dim_kw

        resampled = torch.as_tensor(self._resampling_method(data, **kwarg), 
                                    device=device) 
        dim_order = list(range(len(self.data.shape)))
        dim_order.insert(self.dim, dim_order.pop(0))
        self.resampled = resampled.permute(dim_order).cpu()

        data = self.data
        if self.module != 'torch' and not self.ensured_torch:
            data = np.asarray(data.detach().cpu())
        if self._has_no_dim: theta_hat = self.statistics(data)
        else: theta_hat = self.statistics(data, **{self.dim_kw:self.dim})
        self._theta_hat = torch.as_tensor(theta_hat, device=device
                                          ).unsqueeze(self.dim)

    def _resampling_method(self, data:torch.Tensor, **kwarg):
        raise NotImplementedError(
            '_Resampling should be inherited by a class with defined method.')

    def get_samples(self) -> torch.Tensor: 
        return self.resampled.to(device)

    def get_error(self) -> torch.Tensor: 
        return self.get_samples().std(dim=self.dim, keepdim=True)

    def get_quantile_ci(self, confidence:float=0.95) -> torch.Tensor:
        half_alpha = (1 - confidence) / 2
        q = torch.tensor([half_alpha, 1 - half_alpha], device=device)
        return torch.quantile(self.get_samples(), q, dim=self.dim, keepdim=True
                              ).transpose(0,
                                          self.dim - len(self.data.shape)
                                          ).squeeze()

    def get_reverse_quantile_ci(self, confidence:float=0.95):
        return torch.flip(2 * self._theta_hat - self.get_quantile_ci(confidence),
                          (self.dim,))


class Jackknife(_Resampling):
    def __init__(self, data:torch.Tensor, statistics:callable, dim:int=0, 
                 remove:int=1, dim_kw:Union[str, None]=None, 
                 ensured_torch:bool=False):
        super().__init__(data, statistics, dim=dim, dim_kw=dim_kw, 
                         ensured_torch=ensured_torch, remove=remove)

    def _resampling_method(self, data:torch.Tensor, remove:int) -> torch.Tensor:
        num_sample = data.shape[0]
        idxs = torch.arange(num_sample)
        comb = torch.combinations(idxs, remove)
        sample_idxs = idxs.unsqueeze(0) * torch.ones_like(comb)
        sample_idxs = sample_idxs[sample_idxs != comb
                                  ].view(-1, num_sample - remove).T
        if self._has_no_dim:
            data_ = data[sample_idxs]
            if self.module != 'torch' and not self.ensured_torch:
                data_ = np.asarray(data_.cpu())
            resampled = self.statistics(data_)
        else:
            data_ = data[sample_idxs]
            if self.module != 'torch' and not self.ensured_torch:
                data_ = np.asarray(data_.cpu())
            resampled = self.statistics(data_, **{self.dim_kw:0})
        return resampled


class Bootstrap(_Resampling):
    def __init__(self, data:torch.Tensor, statistics:callable, dim:int=0, 
                 resample_size:Union[int, None]=None, resample_times:int=100000, 
                 buffer:int=int(5e8), dim_kw:Union[str, None]=None, 
                 ensured_torch:bool=False):
        super().__init__(data, statistics, dim=dim, dim_kw=dim_kw, 
                         ensured_torch=ensured_torch, 
                         resample_size=resample_size, resample_times=resample_times,
                         buffer=buffer)

    def _resampling_method(self, data:torch.Tensor, resample_size:Union[int, None], 
                           resample_times:int, buffer:int) -> torch.Tensor:
        if resample_size is None: resample_size = data.shape[0]
        sample_idxs = torch.randint(data.shape[0], 
                                    (resample_size, resample_times))
        dicing_dims = list(range(1, len(data.shape)))
        length = int(np.ceil((buffer / resample_size / resample_times
                             ) ** (1 / len(dicing_dims))))
        if length < 1: length = 1
        bs_shape = [resample_times] + list(data.shape)[1:]
        if self.module != 'torch' and not self.ensured_torch: 
            resampled = np.zeros(bs_shape)
        else: 
            resampled = torch.zeros(bs_shape)
        slices, diced_data = dice(data, length, dicing_dims)
        if self._has_no_dim: 
            for slice_, diced in zip(slices, diced_data):
                diced_ = diced[sample_idxs]
                if self.module != 'torch' and not self.ensured_torch:
                    diced_ = np.asarray(diced_.cpu())
                resampled[slice_] = self.statistics(diced_)
        else:
            for slice_, diced in zip(slices, diced_data):
                diced_ = diced[sample_idxs]
                if self.module != 'torch' and not self.ensured_torch:
                    diced_ = np.asarray(diced_.cpu())
                resampled[slice_] = self.statistics(diced_, **{self.dim_kw:0})
        return resampled

    def get_bca_ci(self, confidence:float=0.95) -> torch.Tensor:
        jk = Jackknife(self.data, self.statistics, self.dim, dim_kw=self.dim_kw,
                       ensured_torch=self.ensured_torch)
        jk_samples = jk.get_samples()
        jk_bias = jk_samples.mean(dim=self.dim, keepdim=True) - jk_samples
        acceleration = (jk_bias ** 3).sum(dim=self.dim, keepdim=True) \
                     / 6 \
                     / ((jk_bias ** 2).sum(dim=self.dim, keepdim=True)) ** (3 / 2)
        gaussian = _Gaussian()
        half_alpha = torch.tensor((1 - confidence) / 2, device=device)
        resampled = self.get_samples()
        quantile = discrete_cdf(resampled, self._theta_hat, dim=self.dim)        
        z_0_hat = gaussian.icdf(quantile)
        z_half_alpha = gaussian.icdf(half_alpha)
        z_q_pos = z_0_hat + z_half_alpha
        z_q_neg = z_0_hat - z_half_alpha
        z = torch.cat([z_0_hat + z_q_pos / (1 - acceleration * z_q_pos),
                       z_0_hat + z_q_neg / (1 - acceleration * z_q_neg)], 
                       dim=self.dim)
        return discrete_quantile(resampled, gaussian.cdf(z), dim=self.dim)


def is_integer(x) -> bool:
    ints = [int, torch.int8, torch.int16, torch.int32, torch.int64,
            np.int8, np.int16, np.int32, np.int64]
    return True if type(x) in ints else False 

def _rectify_dim(data, dim:int) -> int: 
    if not is_integer(dim): 
        raise RuntimeError('dim should be integer of Python, NumPy or Torch')
    if dim < -(num_dims := len(data.shape)) or dim > num_dims - 1:
        raise RuntimeError(f'dim should be in [-{num_dims}, {num_dims - 1}]')
    return dim if dim >= 0 else num_dims + dim

def dice(tensor:torch.Tensor, length:int, 
         dicing_dims:Union[list, tuple]=tuple()) -> tuple:
    if not is_integer(length): raise RuntimeError('length should be integer')
    dicing_dims = tuple(dicing_dims)
    for i in dicing_dims:
        if not is_integer(i): 
            raise RuntimeError('each item of dicing_dims should be integer')
    dims = torch.arange(len(tensor.shape))
    if len(dicing_dims) == 0: dicing_dims = dims
    dice_nums = torch.ceil(torch.tensor(tensor.shape) / length).type(torch.int)
    slicer = list()
    for dim in dims:
        if dim in dicing_dims:
            slice_idxs = (torch.arange(dice_nums[dim]) * length).tolist() + [None]
            slicer.append(
                [slice(i, j) for i, j in zip(slice_idxs[:-1], slice_idxs[1:])])
        else:
            slicer.append([slice(None)])
    slices = list(itertools.product(*slicer))
    diced_tensors = [tensor[slice_] for slice_ in slices]
    return slices, diced_tensors

# TODO: interpolation 부분 다시
# TODO: input validation
def discrete_cdf(samples:torch.Tensor, quantile:torch.Tensor, 
                 dim:int=0, interpolation:str='midpoint') -> torch.Tensor:
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
def discrete_quantile(samples:torch.Tensor, q:torch.Tensor, 
                      dim:int=0) -> torch.Tensor:
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
