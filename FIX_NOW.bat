@echo off
cls
echo.
echo ========================================================
echo   AUTOMATIC FIX: Installing PyTorch Geometric
echo ========================================================
echo.
echo This will install PyTorch Geometric to fix the DGL error.
echo.
echo Using: Python 3.12
echo Location: C:\Users\shiva\AppData\Local\Programs\Python\Python312
echo.
pause
echo.
echo Installing PyTorch Geometric...
echo.

C:\Users\shiva\AppData\Local\Programs\Python\Python312\python.exe -m pip install torch-geometric torch-scatter torch-sparse

echo.
echo ========================================================
echo   Testing Installation
echo ========================================================
echo.

C:\Users\shiva\AppData\Local\Programs\Python\Python312\python.exe -c "from utils.graph_backend import get_available_backends; print('SUCCESS! Available backends:', get_available_backends())"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   SUCCESS! PyTorch Geometric is now installed
    echo ========================================================
    echo.
    echo You can now run your application:
    echo   python ncgn_dashboard.py
    echo.
) else (
    echo.
    echo ========================================================
    echo   Installation completed but verification failed
    echo ========================================================
    echo.
    echo Please check the error messages above.
    echo You may need to run: python fix_backend.py
    echo.
)

pause

