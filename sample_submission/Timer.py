import time


class Timer(object):
    def __init__(self, total_time=0):
        self.total_time = total_time
        self.start_time = None

    @property
    def total(self):
        return self.total_time

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}({self.total_time})'

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # warning: returning True in __exit__
        # WILL SUPPRESS ALL ERRORS
        self.pause()

    def start(self):
        self.start_time = time.perf_counter()

    def pause(self):
        assert self.start_time is not None
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.total_time += duration
        self.start_time = None