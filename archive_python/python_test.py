import sys
with open("D:\\Node_network\\python_test.txt", "w") as f:
    f.write("Python is working\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Python path: {sys.executable}\n")
