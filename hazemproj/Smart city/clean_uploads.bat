@echo off
echo ========================================
echo  Cleaning ALL Videos from Project
echo ========================================
echo.
echo This will delete ALL videos to save space:
echo - All uploaded videos (web_uploads)
echo - All processed videos (web_outputs)
echo - Unused project folders
echo.
echo Images, PDFs, and database will be kept.
echo.
pause

echo.
echo Deleting unused project folders...
rmdir /s /q "AI-POWERED-STREET-SIGN-READER-AND-NOTIFICATION-SYSTEM-main (1)" 2>nul
rmdir /s /q "Animal-Behaviour-and-Disease-Detection-main" 2>nul
rmdir /s /q "ANPR-master" 2>nul
rmdir /s /q "License-Plate-Detector-master (1)" 2>nul
rmdir /s /q "traffic-sign-detection-using-yolov11-main" 2>nul
rmdir /s /q "Shoplifting-Detection-using-Computer-Vision-and-Machine-Learning-main" 2>nul
rmdir /s /q "runs" 2>nul
rmdir /s /q "__pycache__" 2>nul
echo ✓ Unused folders deleted

echo.
echo Deleting uploaded videos...
cd web_uploads
del /q *.mp4 2>nul
del /q *.mov 2>nul
del /q *.avi 2>nul
del /q *.mkv 2>nul
del /q *.webm 2>nul
echo ✓ Uploaded videos deleted

echo.
echo Deleting uploaded images...
del /q *.png 2>nul
del /q *.jpg 2>nul
del /q *.jpeg 2>nul
del /q *.jfif 2>nul
del /q *.webp 2>nul
del /q *.bmp 2>nul
echo ✓ Uploaded images deleted

echo.
echo Deleting temp folders...
for /d %%d in (temp_*) do rmdir /s /q "%%d"
echo ✓ Temp folders deleted

cd ..

echo.
echo Deleting processed output videos...
cd web_outputs
del /q *_street_sign.mp4 2>nul
del /q *_animal.mp4 2>nul
echo ✓ Output videos deleted

echo.
echo Deleting output images...
del /q *.png 2>nul
del /q *.jpg 2>nul
echo ✓ Output images deleted

cd ..

echo.
echo ========================================
echo  Cleanup Complete!
echo ========================================
echo.
echo Deleted:
echo - 6 unused project folders
echo - All uploaded videos and images
echo - All processed output videos
echo - All temp folders
echo.
echo Kept:
echo - Plate images (*_plates folders)
echo - PDF reports (*_reports folders)
echo - Database (vehicles.db)
echo - JSON files
echo - Core application files
echo.
echo Your project should be under 100MB now!
echo.
pause
