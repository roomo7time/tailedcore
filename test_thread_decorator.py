import time

from src.utils import parallel_decorator


@parallel_decorator(max_workers=4)
def print_abc(s):
    print(f"start - {s}")
    time.sleep(5)  # Simulate a task taking 5 seconds
    print(f"end - {s}")


@parallel_decorator(max_workers=8)
def print_def(s):
    print(f"start - {s}")
    time.sleep(5)  # Simulate a task taking 5 seconds
    print(f"end - {s}")


for i in range(100):
    print_abc("abc")
    print_def("def")
