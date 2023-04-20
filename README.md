# ecgr5105_Project

Because of updates to PyTorch, you need to replace `from torch.autograd.gradcheck import zero_gradients` from `DeepFool/Python/deepfool.py` with 

```
def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)
```