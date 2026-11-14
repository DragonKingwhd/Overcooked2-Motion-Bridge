@echo off
@chcp 936 >nul
title Overcooked 2 Motion Controller
cd /d "%~dp0"
echo ========================================================
echo   Overcooked 2 Dual Motion Controller
echo ========================================================
echo.
python motion_overcooked.py
pause
