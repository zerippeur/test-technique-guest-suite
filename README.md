# test-technique-guest-suite

Projet pour un test technique à réaliser dans le cadre d'un processus de recrutement pour la société Guest Suite.   

## INSTRUCTIONS 
Les instructions sont disponibles dans le fichier test_technique_GuestSuite.pdf situé dans le répertoire data.   

### Contexte du Test Technique :
Construire un modèle ML capable de prédire la note associée à un avis en se basant uniquement sur le texte de cet avis.   
Les avis fournis sont des avis déposés par des clients sur des hôtels.   
L'objectif est de vérifier si la note attribuée par le client correspond bien au contenu de son avis.   
Cette prédiction nous permettra d'identifier des incohérences potentielles et d'améliorer la fiabilité des évaluations des hôtels.      

### Données :
Les données sont téléchargeables avà l'adresse suivante :   
https://drive.google.com/file/d/1TrnRqGvOoif7kLgenEwIyf0JlQlheMCx/view?usp=sharing

- train.csv contient les colonnes suivantes :
    - establishment_id, review_id, review_text, global_rate.
- test.csv contient les colonnes suivantes :
    - establishment_id, review_id, review_text.  

### Livrable attendu :
Projet Python avec une interface en ligne de commande qui permet :
1. De générer un modèle.
2. D'utiliser un modèle pour faire des prédictions sur un dataset de test.
Le fichier CSV de prédictions.

## ANALYSE PRÉALABLE

Une analyse préalable a été réalisée afin de définir une méthodologie de ML adaptée au problème.   
Elle est consultable sous la forme d'un notebook intitulé "data_exploration.ipynb".   
Lors de cette analyse exploratoire, les données ont été nettoyées et enregistrées sous format .pkl.gz.   

## MISE EN PLACE DU PROJET

### Prérequis

- Python 3.11 ou plus.

### Instructions d'installation

#### Cloner le dépôt et se placer dans le répertoire :   
```bash
git clone https://github.com/zerippeur/test-technique-guest-suite.git
cd test-technique-guest-suite
```

#### Exécuter le script d'installation.   
- Pour Linux/Mac :
```bash
./install.sh
```
- Pour Windows :
```bash
./install.bat
```

#### Exécution.   
1. Activer l'environnement virtuel.   
- Pour Linux/Mac :
```bash
source venv/bin/activate
```
- Pour Windows :
```bash
venv\Scripts\activate
```

2. Lancer l'entraînement du modèle.   

```bash
python ./test_technique_guest_suite/model_fit.py
```

3. Lancer la prédiction du modèle.   
 
```bash
python ./test_technique_guest_suite/model_predict.py
```

Le fichier de prédiction se sauvegarde automatiquement sous format .csv dans le dossier data.   

## AMÉLIORATIONS POSSIBLES

Il existe différentes pistes d'améliorations :
- Approfondir le nettoyage des données textuelles.
- Utiliser d'autres modèles d'extraction de features (fasttext, BERT, USE) ou réseau de neurones (GRU, LSTM).
- Essayer des visualisations en faible dimensions (t-SNE) en affichant la typologie NPS ou la note.
- Optimiser les hyperparamètres du modèle et utiliser un score customisé (minimiser les erreurs de classification entre les catégories les plus extrêmes du NPS par exemple).
- Utiliser le ARI score en plus de l'accuracy.
- Enregistrer les runs grâce à mlflow et faire du model serving (dagshub, mlflow).
- Créer une api de prédiction et un dashboard permettant de requêter cette api pour visualiser les prédictions et afficher les textes et les notes réelles.

## PROJET GIT

https://github.com/zerippeur/test-technique-guest-suite