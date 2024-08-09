@echo off

REM Fonction pour vérifier la version de Python
:check_python_version
python --version 2>&1 | findstr /C:"3.11" > nul
if %errorlevel% neq 0 (
    echo Python 3.11 ou supérieur est requis. Vous utilisez une version de Python inférieure.
    echo Veuillez installer Python 3.11 ou supérieur.
    exit /b 1
)

REM Créer un environnement virtuel
python -m venv venv

REM Activer l'environnement virtuel
call venv\Scripts\activate

REM Installer les dépendances
pip install -r requirements.txt

REM Désactiver l'environnement virtuel
deactivate

echo Installation terminée. Vous pouvez maintenant activer l'environnement virtuel et exécuter le script principal.