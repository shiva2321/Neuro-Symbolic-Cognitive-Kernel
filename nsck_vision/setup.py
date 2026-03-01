from setuptools import setup, find_packages

setup(
    name="nsck-vision",
    version="1.0.0",
    description="NSCK Universal Pretrained Model Absorber (NSCK-UPMA)",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "Pillow>=9.0.0",
            "flask>=2.3.0",
            "flask-socketio>=5.3.0",
            "onnxruntime>=1.16.0",
        ],
        "transformers": [
            "transformers>=4.30.0",
            "timm>=0.9.0",
            "open_clip_torch>=2.20.0",
        ],
    },
    python_requires=">=3.8",
)
