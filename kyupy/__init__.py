"""This package provides tools for high-performance processing and validation
of non-hierarchical VLSI circuits to aid rapid prototyping of research code
in the fields of VLSI test, diagnosis and reliability.
"""

import time
import importlib.util


class Log:
    def __init__(self):
        self.start = time.perf_counter()
        self.logfile = None

    def log(self, level, message):
        t = time.perf_counter() - self.start
        if self.logfile is None:
            print(f'{t:011.3f} {level} {message}')
        else:
            self.logfile.write(f'{t:011.3f} {level} {message}\n')
            self.logfile.flush()

    def info(self, message): self.log('-', message)

    def warn(self, message): self.log('W', message)

    def error(self, message): self.log('E', message)


log = Log()


class MockNumba:
    @staticmethod
    def njit(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


class MockCuda:

    def __init__(self):
        self.x = 0
        self.y = 0

    def jit(self, device=False):
        _ = device  # silence "not used" warning
        outer = self

        def make_launcher(func):
            class Launcher(object):
                def __init__(self, funcc):
                    self.func = funcc

                def __call__(self, *args, **kwargs):
                    # print(f'device func call {self.func.__name__}')
                    return self.func(*args, **kwargs)

                def __getitem__(self, item):
                    grid_dim, block_dim = item
                    # print(f'kernel call {self.func.__name__} grid_dim:{grid_dim} block_dim:{block_dim}')

                    def inner(*args, **kwargs):
                        for grid_x in range(grid_dim[0]):
                            for grid_y in range(grid_dim[1]):
                                for block_x in range(block_dim[0]):
                                    for block_y in range(block_dim[1]):
                                        outer.x = grid_x * block_dim[0] + block_x
                                        outer.y = grid_y * block_dim[1] + block_y
                                        self.func(*args, **kwargs)
                    return inner
            return Launcher(func)

        return make_launcher

    @staticmethod
    def to_device(array, to=None):
        if to is not None:
            to[...] = array
            return to
        return array.copy()

    def synchronize(self):
        pass

    def grid(self, dims):
        _ = dims  # silence "not used" warning
        return self.x, self.y


if importlib.util.find_spec('numba') is not None:
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.error import CudaSupportError
    try:
        list(numba.cuda.gpus)
        from numba import cuda
    except CudaSupportError:
        log.warn('Cuda unavailable. Falling back to pure python')
        cuda = MockCuda()
else:
    numba = MockNumba()
    cuda = MockCuda()
    log.warn('Numba unavailable. Falling back to pure python')



