import time as time_module

def f():
    for i in range(10**7):
        1 + 1
    return i


start_time = time_module.monotonic()
x = f()
end_time = time_module.monotonic()
duration = end_time - start_time
print(x)
print(duration)
