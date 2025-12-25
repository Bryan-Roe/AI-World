@echo off
REM health-check.bat - Monitor server and Ollama health

echo Health Check
echo ============
echo.

REM Check server
echo [1/2] Checking web server...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:3000/health; Write-Host \"+ Server OK: $($r.Content)\"; exit 0 } catch { Write-Host \"- Server FAILED\"; exit 1 }"
if errorlevel 1 goto :server_fail
goto :server_ok

:server_fail
echo X Server not responding
goto :check_ollama

:server_ok
echo.

:check_ollama
REM Check Ollama
echo [2/2] Checking Ollama...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:11434/api/tags; $json = $r.Content | ConvertFrom-Json; if ($json.models.Count -eq 0) { Write-Host 'W Ollama OK but no models. Run: ollama pull gpt-oss-20'; exit 0 } else { Write-Host \"+ Ollama OK. Models: $($json.models | % {$_.name})\"; exit 0 } } catch { Write-Host 'X Ollama FAILED'; exit 1 }"

echo.
echo Done.
pause
