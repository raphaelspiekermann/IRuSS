import logging
import time
from functools import wraps


def timeit(
    func=None,
    /,
    *,
    n_iter: int = 1,
    n_warmup_iter: int = 0,
    print_input: bool = False,
    print_output: bool = False,
    use_print: bool = False,
):
    def decorator_timeit(func):
        @wraps(func)
        def wrapper_timeit(*args, **kwargs):
            print_fn = print if use_print else logging.info
            for _ in range(n_warmup_iter):
                result = func(*args, **kwargs)
            start_time = time.time()
            for _ in range(n_iter):
                result = func(*args, **kwargs)
            end_time = time.time()
            if print_input:
                print_fn(f"@Timeit({func.__name__}) -> Input: {args}, {kwargs}")
            if print_output:
                print_fn(f"@Timeit({func.__name__}) -> Output: {result}")
            print_fn(
                f"@Timeit({func.__name__}) -> Execution time: {end_time - start_time} seconds"
            )
            return result

        return wrapper_timeit

    if func is None:  # @timeit() -> @timeit
        return decorator_timeit

    return decorator_timeit(func)
