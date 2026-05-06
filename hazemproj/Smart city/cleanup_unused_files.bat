@echo off
echo ========================================
echo  Smart City Project Cleanup Script
echo ========================================
echo.
echo This will delete unused folders and files to reduce project size.
echo.
pause

echo Deleting unused project folders...
rmdir /s /q "AI-POWERED-STREET-SIGN-READER-AND-NOTIFICATION-SYSTEM-main (1)"
rmdir /s /q "Animal-Behaviour-and-Disease-Detection-main"
rmdir /s /q "ANPR-master"
rmdir /s /q "License-Plate-Detector-master (1)"
rmdir /s /q "traffic-sign-detection-using-yolov11-main"
rmdir /s /q "runs"
rmdir /s /q "__pycache__"

echo.
echo Deleting old web uploads (keeping structure)...
del /q "web_uploads\*.mp4" 2>nul
del /q "web_uploads\*.mov" 2>nul
del /q "web_uploads\*.png" 2>nul
del /q "web_uploads\*.jpg" 2>nul
del /q "web_uploads\*.jfif" 2>nul
for /d %%d in ("web_uploads\temp_*") do rmdir /s /q "%%d"

echo.
echo Deleting old web outputs (keeping recent ones)...
REM Keep only the last 5 outputs, delete older ones
REM You can manually delete specific tokens if needed

echo.
echo ========================================
echo  Cleanup Complete!
echo ========================================
echo.
echo Deleted:
echo - Unused project folders (5 folders)
echo - Test and debug files
echo - Old uploads
echo - Python cache files
echo.
echo Your project should be much lighter now!
echo.
pause
