@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
%*
if %errorlevel% neq 0 exit /b %errorlevel%
echo Check Successful
