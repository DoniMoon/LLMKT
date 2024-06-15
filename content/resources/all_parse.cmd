@echo off
setlocal enabledelayedexpansion

REM Get the directory of the script itself
set "script_dir=%~dp0"

for /R "%script_dir%" %%f in (*.swf) do (
    echo Processing: %%f

    REM Create a temporary file to store the output for parsing
    swfextract.exe %%f > temp.txt

    REM Extract JPEG IDs
    for /f "tokens=*" %%a in ('findstr /C:"[-j]" temp.txt') do (
        set "line=%%a"
        set "line=!line:*ID(s)=! "
        for %%i in (!line!) do (
            swfextract.exe -j %%i -o "%%~dpnf_%%i.jpeg" %%f
        )
    )

    REM Extract PNG IDs
    for /f "tokens=*" %%a in ('findstr /C:"[-p]" temp.txt') do (
        set "line=%%a"
        set "line=!line:*ID(s)=! "
        for %%i in (!line!) do (
            swfextract.exe -p %%i -o "%%~dpnf_%%i.png" %%f
        )
    )
)

del temp.txt
echo Done.