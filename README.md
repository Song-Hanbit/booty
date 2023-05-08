# booty
<ins>Boo</ins>tstrap resampling w/ <ins>t</ins>orch at p<ins>y</ins>thon :shoe::fire::snake:

This is a clone-coded script of `scipy.stats.bootstrap` with some twists.

# Dependencies
This script was tested with the version of `python` packages:
```
torch 2.0.0+cu117
numpy 1.23.4
```
with CUDA-toolkit version of 11.7.

# How to use
The `Bootstrap` class has following arguments.

>`Input:`
>```python
>from booty.booty import *
>Bootstrap?
>```
>
>`Printed:`
>```
>device: cuda
>Init signature:
>Bootstrap(
>    data,
>    statistics,
>    dim=0,
>    resample_size=None,
>    resample_times=10000,
>    dim_kw=None,
>    ensured_torch=False,
>)
>Docstring:      <no docstring>
>File:           ~/booty/booty.py
>Type:           type
>Subclasses:     
>```
>
>`data` takes data as `torch.Tensor`.
>
>`statistics` takes a name of method that return statistical parameters like `torch.mean`.
>
>`dim` is the dimension to reduce using `statistics`.
>
>`resample_size` is the number of sample per resampled samples. If it is `None`, it will be automatically assigned as `data.shape[dim]`.
>
>`resample_times` is the number of resampling.
>
>`dim_kw` is the keyword argument of dimension like `dim` for `torch` or `axis` for `numpy`. If it is `None`, it will be automatically assigned as `'dim'` for `torch` and `'axis'` for `numpy`.
>
>`ensured_torch` is a boolean that says `statistics` is `torch` based method even though its `__module__` is not `'torch'`.

Therefore, you need to obtain some data as `torch.Tensor`.

>`Input:`
>```python
>import torch
>data = torch.normal(0, 1, (40, 50, 60))
>data.shape
>```
>
>`Printed:`
>```
>torch.Size([40, 50, 60])
>```

The instance of `Bootstrap` will perform bootstrapping at the initialization. The instance have some methods and all the values of interest are at the `dim` of outputs.:
 
>`get_samples`: returns samples of the bootstrapped statistics.
>
>`get_error`: returns the bootstrapped error of the statistics.
>
>`get_quantile_ci(confidence=0.95)`: returns lower and upper bound of `statistics`' confidence interval according to the `confidence` level using the quantile of bootstrapped samples.
>
>`get_reverse_quantile_ci(confidence=0.95)`: same as `get_quantile_ci` using the 'reverse percentile' method.
>
>`get_bca_ci(confidence=0.95)`: same as `get_quantile_ci` using the 'bias-corrected and accelerated' method.
___
>`Input:`
>```python
>b = Bootstrap(data, torch.mean, dim=0)
>print(b.get_samples().shape, b.get_error().shape, b.get_bca_ci().shape, sep='\n')
>```
>
>`Printed:`
>```
>torch.Size([10000, 50, 60])
>torch.Size([1, 50, 60])
>torch.Size([2, 50, 60])
>```

Reduction of the another dimension is also available by assigning `dim`.

>`Input:`
>```python
>b = Bootstrap(data, torch.mean, dim=-1)
>print(b.get_samples().shape, b.get_error().shape, b.get_bca_ci().shape, sep='\n')
>```
>
>`Printed:`
>```
>torch.Size([40, 50, 10000])
>torch.Size([40, 50, 1])
>torch.Size([40, 50, 2])
>```

You can plug the another method in the `statistics` argument. e.g. `numpy.mean`

>`Input:`
>```python
>import numpy as np
>b = Bootstrap(data, np.mean, dim=-1)
>print(b.get_samples().shape, b.get_error().shape, b.get_bca_ci().shape, sep='\n')
>```
>
>`Printed:`
>```
>torch.Size([40, 50, 10000])
>torch.Size([40, 50, 1])
>torch.Size([40, 50, 2])
>```

>`Input:`
>```python
>def median(data, dim=0): 
>    return torch.median(data, dim=dim)[0]
>    
>b = Bootstrap(data, median, dim=-1, ensured_torch=True)
>print(b.get_samples().shape, b.get_error().shape, b.get_bca_ci().shape, sep='\n')
>```
>
>`Printed:`
>```
>torch.Size([40, 50, 10000])
>torch.Size([40, 50, 1])
>torch.Size([40, 50, 2])
>```
