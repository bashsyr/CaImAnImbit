import time
import functools
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        func_name = func.__name__
        print(f"Starting {func_name}")

        t1 = time.perf_counter()
        output = func(*args, **kwargs)
        t2 = time.perf_counter()

        print(f"Total time for {func_name}: {t2 - t1:.3f} s\n")
        return output

    return wrapper