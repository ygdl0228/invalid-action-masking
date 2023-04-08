# @Time    : 2023/4/5 22:20
# @Author  : ygd
# @FileName: test-2.py
# @Software: PyCharm

import torch

a = torch.Tensor([[7.4409e+00, 3, 2, 1],
                  [7.4409e+00, 3, 2, 1],
                  [7.4161e+00, 1.0295e+01, 4, 2],
                  [7.4332e+00, 1.0321e+01, 5, 7],
                  [7.4332e+00, 1.0321e+01, 2, 4]])
print(a.max(1)[1])
