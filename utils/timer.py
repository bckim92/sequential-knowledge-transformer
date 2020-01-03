from timeit import default_timer


class Timer(object):
    """Context manager for timing code blocks

    To use:
    >>> with Timer(verbose=True) as t:
    >>>     # some code here
    >>> print(t.elapsed)
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.timer = default_timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs
        if self.verbose:
            print(f'elapsed time: {self.elapsed:.3f} sec')
