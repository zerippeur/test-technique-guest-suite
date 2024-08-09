#!/bin/bash

# Fonction pour vérifier la version de Python
check_python_version() {
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    if [[ "$python_version" < "3.11" ]]; then
        echo "Python 3.11 ou supérieur est requis. Vous utilisez Python $python_version."
        echo "Veuillez installer Python 3.11 ou supérieur."
        exit 1
    fi
}

# Vérifier la version de Python
check_python_version

# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Désactiver l'environnement virtuel
deactivate

echo "Installation terminée. Vous pouvez maintenant activer l'environnement virtuel et exécuter le script principal."