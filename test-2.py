# @Time    : 2023/4/5 22:20
# @Author  : ygd
# @FileName: test-2.py
# @Software: PyCharm

import numpy as np
import multiprocessing
process_num=4
pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
print(pipe_dict[0][1])