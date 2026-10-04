@echo off
rem Build sr-native. Requires third_party/ to be staged first:
rem
rem     powershell -NoProfile -ExecutionPolicy Bypass -File setup-third-party.ps1
rem
rem This exists as a batch file for one reason: MSVC's environment is only set up
rem by vcvars64.bat, and a batch file is the only thing PowerShell and build.rs can
rem both call without reimplementing that detection. CMake and Ninja ship inside
rem Visual Studio and are not on PATH.

setlocal
if "%SR_NATIVE_VS%"=="" set SR_NATIVE_VS=C:\Program Files\Microsoft Visual Studio\18\Community
if not exist "%SR_NATIVE_VS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo VCVARS_MISSING: %SR_NATIVE_VS%\VC\Auxiliary\Build\vcvars64.bat
    echo Set SR_NATIVE_VS to the Visual Studio installation directory.
    exit /b 1
)
call "%SR_NATIVE_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )

rem %~dp0 ends with a backslash, which would escape the closing quote in the cmake
rem command below. Drop it.
set SRC=%~dp0
set SRC=%SRC:~0,-1%
set BUILD=%SRC%\build

where ninja >nul 2>&1
if errorlevel 1 (
    set GEN=-G "Visual Studio 18 2026"
) else (
    set GEN=-G Ninja
)

cmake -S "%SRC%" -B "%BUILD%" %GEN% -DCMAKE_BUILD_TYPE=Release >nul
if errorlevel 1 ( echo CONFIGURE_FAILED & exit /b 2 )

cmake --build "%BUILD%" --config Release
if errorlevel 1 ( echo BUILD_FAILED & exit /b 3 )

echo NATIVE_BUILD_OK
exit /b 0
