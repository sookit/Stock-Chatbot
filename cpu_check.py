import psutil

memory = psutil.virtual_memory()
print(f"Total memory: {memory.total / (1024 ** 3):.2f} GB")
print(f"Available memory: {memory.available / (1024 ** 3):.2f} GB")
print(f"Used memory: {memory.used / (1024 ** 3):.2f} GB")
