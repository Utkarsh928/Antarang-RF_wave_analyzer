@echo off
REM Build all three C++ DLLs for Signal Analyzer Pro
REM Uses MSVC 2022 Build Tools

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1

set SRC=%~dp0
set OUT=%SRC%..\

echo Building signal_features.dll ...
cl /O2 /LD /EHsc /W3 "%SRC%signal_features.cpp" /Fe:"%OUT%signal_features.dll" /link /MACHINE:X64
if %ERRORLEVEL% NEQ 0 ( echo FAILED: signal_features.dll & exit /b 1 )
echo   OK

echo Building viterbi.dll ...
cl /O2 /LD /EHsc /W3 "%SRC%viterbi.cpp" /Fe:"%OUT%viterbi.dll" /link /MACHINE:X64
if %ERRORLEVEL% NEQ 0 ( echo FAILED: viterbi.dll & exit /b 1 )
echo   OK

echo Building fast_correlator.dll ...
cl /O2 /LD /EHsc /W3 "%SRC%fast_correlator.cpp" /Fe:"%OUT%fast_correlator.dll" /link /MACHINE:X64
if %ERRORLEVEL% NEQ 0 ( echo FAILED: fast_correlator.dll & exit /b 1 )
echo   OK

REM Clean up obj and lib files (keep only .dll)
del /Q "%SRC%*.obj" 2>nul
del /Q "%OUT%*.lib" 2>nul
del /Q "%OUT%*.exp" 2>nul

echo.
echo All DLLs built successfully.
echo Output: %OUT%signal_features.dll
echo         %OUT%viterbi.dll
echo         %OUT%fast_correlator.dll
