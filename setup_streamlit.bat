@echo off

REM Create the .streamlit directory in the user's home folder
mkdir %USERPROFILE%\.streamlit

REM Write the configuration to config.toml
echo [server]> %USERPROFILE%\.streamlit\config.toml
echo port = %PORT%>> %USERPROFILE%\.streamlit\config.toml
echo enableCORS = false>> %USERPROFILE%\.streamlit\config.toml
echo headless = true>> %USERPROFILE%\.streamlit\config.toml
