import sys
import platform
from importlib.metadata import version, PackageNotFoundError

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm",
]

print(f"Python {sys.version}")
print(f"Platform: {platform.platform()}")
for pkg in PACKAGES:
    try:
        print(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg}: NOT INSTALLED")