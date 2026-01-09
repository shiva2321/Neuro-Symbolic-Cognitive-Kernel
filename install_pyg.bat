@echo off
echo ========================================
echo Installing PyTorch Geometric
echo ========================================
echo.

echo This will install PyTorch Geometric to replace the corrupted DGL installation.
echo.

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'CPU only')"

echo.
echo Installing PyTorch Geometric...
echo.

pip install torch-geometric torch-scatter torch-sparse torch-cluster

echo.
echo ========================================
echo Installation Complete
echo ========================================
echo.
echo Testing imports...
python -c "import torch_geometric; print('PyG version:', torch_geometric.__version__); print('SUCCESS: PyTorch Geometric is working!')"

echo.
echo You can now run your application!
echo.
pause

