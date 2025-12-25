@echo off
echo Building EcoScan.exe...
echo.

:: Clean previous builds
rmdir /s /q build dist
del /q *.spec

:: Build with PyInstaller
:: --onedir: Folder based distribution
:: --windowed: No console (GUI only) - Change to --console if you want debugging
:: --icon: (Optional, add if you have an .ico file)
:: --add-data: Bundling model and ml package
pyinstaller --noconfirm --onedir --console --name "EcoScan" ^
    --add-data "coral_bleaching_resnet18.pt;." ^
    --add-data "ml;ml" ^
    --hidden-import "PIL._tkinter_finder" ^
    desktop_app.py

echo.
echo Build Complete!
echo You can find the app in: dist\EcoScan\EcoScan.exe
pause
