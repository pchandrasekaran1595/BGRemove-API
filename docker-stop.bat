start /MAX cmd /c "title STOP-API-SERVER && cls && docker stop bgremove-container && docker rm -f bgremove-container && timeout /t 10 /nobreak"