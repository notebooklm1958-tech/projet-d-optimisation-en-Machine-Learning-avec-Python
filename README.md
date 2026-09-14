# projet-d-optimisation-en-Machine-Learning-avec-Python
# Optimisation des prédictions d’un modèle de Machine Learning

## 1. Objectif du projet

L’objectif de ce projet est de construire et d’optimiser un modèle de **Machine Learning** capable de prédire la superficie brûlée lors d’un incendie de forêt.

Le jeu de données utilisé contient des observations provenant d’incendies de forêt au Portugal. Chaque observation contient différentes informations géographiques et météorologiques, notamment :

* les coordonnées géographiques `X` et `Y` ;
* le jour et le mois ;
* la température ;
* l’humidité relative ;
* le vent ;
* les précipitations ;
* différents indicateurs météorologiques tels que `FFMC`, `DMC` et `DC` ;
* la variable cible `area`, représentant la superficie brûlée.

Le problème est donc un problème de **régression**, puisque la variable à prédire est une valeur numérique continue.

L'objectif général est de construire un modèle aussi performant que possible en suivant une démarche systématique :

1. explorer les données ;
2. nettoyer les données ;
3. transformer les variables ;
4. traiter les valeurs manquantes ;
5. séparer les données d’entraînement et de test ;
6. sélectionner les variables pertinentes ;
7. construire un modèle de régression ;
8. régulariser le modèle ;
9. comparer les performances ;
10. déterminer si le modèle apporte réellement une amélioration par rapport à une référence simple.

---

# 2. Importation des bibliothèques

Le projet utilise principalement **pandas**, **NumPy** et **scikit-learn**.

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

D’autres outils de scikit-learn seront importés au fur et à mesure du projet.

---

# 3. Chargement des données

Les données sont stockées dans un fichier CSV appelé `fires.csv`.

```python
fires = pd.read_csv("fires.csv")
```

Pour examiner rapidement les premières observations :

```python
fires.head()
```

Le jeu de données contient **517 observations**. Certaines colonnes nécessitent un nettoyage avant de pouvoir être utilisées directement par un modèle de Machine Learning.

---

# 4. Exploration initiale des données

Une première étape importante consiste à examiner la structure du DataFrame.

```python
fires.info()
```

La méthode `info()` permet notamment de connaître :

* le nombre de lignes ;
* les colonnes ;
* le nombre de valeurs non nulles ;
* les types de données.

Cette étape permet d'identifier immédiatement deux problèmes importants :

* certaines colonnes contiennent des **valeurs manquantes** ;
* certaines variables sont de type **texte** (`object`) alors que les modèles de Machine Learning nécessitent généralement des valeurs numériques.

---

# 5. Suppression des colonnes inutiles

La colonne `Unnamed: 0` correspond à un ancien index provenant du fichier CSV. Elle n'apporte pas d'information utile au modèle.

La colonne `day` est également considérée comme peu utile pour ce projet.

La colonne `month` sera transformée puis représentée sous forme de saisons.

Après cette transformation, les anciennes colonnes inutiles pourront être supprimées.

```python
fires = fires.drop(columns=["Unnamed: 0"])
```

---

# 6. Transformation de la variable `month`

La colonne `month` contient des chaînes de caractères représentant les mois :

```text
jan
feb
mar
...
dec
```

Un modèle de Machine Learning ne peut pas utiliser directement ces chaînes de caractères.

Une première transformation consiste donc à associer un nombre à chaque mois :

```text
jan → 1
feb → 2
mar → 3
...
dec → 12
```

Cependant, cette représentation numérique présente un problème.

Si janvier vaut `1` et décembre `12`, un modèle peut interpréter décembre comme une valeur beaucoup plus grande que janvier.

Or, **12 n'est pas douze fois plus important que 1** dans le contexte des mois.

Il est donc préférable de représenter les mois autrement.

---

# 7. Création de la variable `season`

Une solution consiste à regrouper les mois en quatre saisons :

* printemps ;
* été ;
* automne ;
* hiver.

On crée alors une nouvelle colonne `season` à partir du mois.

L'étape suivante consiste à utiliser un encodage **one-hot** afin de transformer la variable catégorielle en variables numériques.

On obtient alors des colonnes telles que :

```text
season_spring
season_summer
season_winter
```

Chaque colonne contient des valeurs `0` ou `1`.

Par exemple :

```text
season_spring = 1
```

signifie que l'observation appartient au printemps.

Une catégorie peut être supprimée avec `drop_first=True`. Dans ce cas, l'une des saisons sert implicitement de catégorie de référence.

---

# 8. Analyse de la variable cible `area`

La variable `area` représente la superficie brûlée.

Avant de construire le modèle, il est important d'étudier sa distribution.

Un histogramme permet de visualiser cette distribution.

```python
import matplotlib.pyplot as plt

plt.hist(fires["area"])
plt.xlabel("Superficie brûlée")
plt.ylabel("Nombre d'observations")
plt.title("Distribution de la superficie brûlée")
plt.show()
```

La distribution de `area` est fortement asymétrique :

* beaucoup d'observations correspondent à une superficie faible ou nulle ;
* quelques observations présentent des superficies très importantes.

La distribution possède donc une **longue queue**.

---

# 9. Transformation de la variable cible

Lorsque la variable cible est fortement asymétrique, différentes transformations peuvent être testées.

Par exemple :

### Transformation logarithmique

```python
np.log1p(fires["area"])
```

### Transformation par racine carrée

```python
np.sqrt(fires["area"])
```

### Transformation par racine cubique

```python
np.cbrt(fires["area"])
```

Ces transformations peuvent parfois améliorer les performances d'un modèle.

Dans ce jeu de données particulier, les différentes transformations testées n'ont toutefois pas amélioré suffisamment les résultats. La variable `area` originale a donc été conservée.

---

# 10. Préparation finale des variables

Après avoir créé les variables représentant les saisons, les colonnes devenues inutiles peuvent être supprimées.

Les variables supprimées comprennent notamment :

```text
Unnamed: 0
day
month
```

L'objectif est d'obtenir un ensemble de données essentiellement numérique, utilisable par les algorithmes de Machine Learning.

---

# 11. Analyse des corrélations

La corrélation permet d'obtenir une première indication sur les relations entre les variables.

```python
fires.corr(numeric_only=True)
```

Cette analyse montre que les corrélations entre les variables explicatives et `area` sont généralement faibles.

Cela constitue déjà un premier indice : les variables disponibles semblent avoir un pouvoir prédictif limité pour expliquer directement la superficie brûlée.

Il faut cependant éviter de conclure qu'une variable est inutile uniquement parce que sa corrélation linéaire est faible. Une relation non linéaire peut exister même lorsque la corrélation linéaire est faible.

---

# 12. Séparation entre variables explicatives et variable cible

Dans un problème de Machine Learning supervisé, on distingue :

* `X` : les variables explicatives ;
* `y` : la variable cible à prédire.

Ici :

```python
X = fires.drop(columns=["area"])
y = fires["area"]
```

`X` contient donc les informations utilisées pour effectuer la prédiction.

`y` contient la superficie réellement brûlée.

---

# 13. Séparation des données d'entraînement et de test

Les données doivent être séparées en deux groupes :

### Données d'entraînement

Elles servent à construire le modèle.

### Données de test

Elles servent uniquement à évaluer le modèle sur des observations qu'il n'a jamais utilisées pendant son apprentissage.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Une partie d'environ 20 % des données peut être utilisée pour le test.

Le paramètre `random_state` permet de rendre la séparation reproductible.

---

# 14. Pourquoi isoler les données de test ?

Les données de test doivent rester isolées pendant le processus de construction du modèle.

Il ne faut pas utiliser les informations du jeu de test pour :

* calculer les paramètres d'une transformation ;
* effectuer une imputation ;
* sélectionner les variables ;
* optimiser le modèle.

Sinon, on risque de créer une **fuite de données** (*data leakage*).

Le score obtenu sur les données de test pourrait alors sembler meilleur qu'il ne l'est réellement.

---

# 15. Traitement des valeurs manquantes

Les algorithmes de Machine Learning utilisés ici ne fonctionnent pas correctement avec certaines valeurs manquantes.

Il faut donc effectuer une **imputation**.

Une méthode utilisée dans le projet est l'algorithme **K-Nearest Neighbors (KNN)**.

L'idée générale est de rechercher des observations similaires et d'utiliser leurs informations pour estimer une valeur manquante.

Par exemple, pour une observation contenant une température mais dont une autre variable est manquante, l'algorithme recherche des observations proches selon les autres caractéristiques disponibles.

---

# 16. Imputation avec KNN

L'imputation doit être ajustée uniquement sur les données d'entraînement.

Le principe est :

```text
X_train
   ↓
fit de l'imputer
   ↓
transformation de X_train
   ↓
transformation de X_test
```

Le jeu de test ne doit pas participer au calcul des valeurs utilisées pour l'imputation.

Cela permet de conserver une véritable évaluation indépendante.

---

# 17. Analyse des valeurs aberrantes

Les **valeurs aberrantes** (*outliers*) sont des observations qui s'éloignent fortement de la majorité des données.

Elles peuvent être étudiées avec des diagrammes en boîte (*boxplots*).

```python
fires.boxplot()
plt.xticks(rotation=90)
plt.show()
```

Une autre méthode consiste à utiliser l'**IQR** (*Interquartile Range*).

L'IQR est calculé à partir des premier et troisième quartiles :

```text
IQR = Q3 - Q1
```

Les observations situées très loin de l'intervalle central peuvent être considérées comme des valeurs aberrantes.

Dans le jeu de données étudié, certaines variables contiennent plusieurs valeurs aberrantes, notamment :

* `FFMC` ;
* `DMC` ;
* d'autres variables météorologiques.

---

# 18. Faut-il supprimer les valeurs aberrantes ?

Il ne faut pas supprimer automatiquement toutes les valeurs aberrantes.

Dans un jeu de données concernant les incendies de forêt, des situations extrêmes peuvent être particulièrement importantes.

Une observation correspondant à un incendie exceptionnel peut précisément contenir une information utile pour la prédiction.

De plus, le jeu de données ne contient qu'environ 500 observations.

Supprimer un grand nombre d'observations pourrait donc réduire considérablement la quantité d'information disponible.

Dans ce projet, les valeurs aberrantes ont finalement été conservées.

---

# 19. Standardisation des variables

Les variables explicatives peuvent avoir des échelles très différentes.

Par exemple :

* certaines variables peuvent avoir des valeurs décimales ;
* d'autres peuvent être exprimées en dizaines ;
* d'autres encore peuvent atteindre plusieurs centaines ou milliers.

La standardisation permet de placer les variables sur une échelle comparable.

Le principe général est :

$$
z = \frac{x-\mu}{\sigma}
$$

où :

* \(x\) est la valeur originale ;
* \(\mu\) est la moyenne ;
* \(\sigma\) est l'écart-type.

Cette étape est particulièrement importante pour certaines méthodes de Machine Learning et pour les modèles régularisés.

---

# 20. Sélection des variables

Le jeu de données contient plusieurs variables.

Cependant, toutes ne contribuent pas nécessairement à la prédiction de `area`.

Utiliser trop de variables peut :

* augmenter la complexité du modèle ;
* ajouter du bruit ;
* rendre le modèle plus difficile à interpréter.

Une méthode automatique utilisée ici est le **Sequential Feature Selector** de scikit-learn.

Cette méthode permet de rechercher progressivement les variables les plus intéressantes.

---

# 21. Sélection progressive : Forward Selection

La sélection progressive (*forward selection*) commence avec un petit nombre de variables.

Elle ajoute ensuite progressivement les variables qui améliorent le plus le modèle.

Le projet teste notamment :

* 2 variables ;
* 4 variables ;
* 6 variables.

La sélection utilise une régression linéaire et une validation croisée en 5 parties (*5-fold cross-validation*).

Les résultats montrent notamment que :

### 2 variables

```text
X
DMC
```

RMSE ≈ 108,909.

### 4 variables

```text
X
DMC
DC
Temperature
```

RMSE ≈ 108,740.

### 6 variables

```text
X
FFMC
DMC
DC
Temperature
Season
```

L'erreur diminue légèrement, mais l'amélioration reste faible.

---

# 22. Sélection régressive : Backward Selection

Une deuxième approche consiste à effectuer une sélection descendante (*backward selection*).

Au lieu de commencer avec peu de variables et d'en ajouter progressivement, on commence avec toutes les variables puis on retire progressivement celles qui semblent les moins utiles.

Cette méthode produit des combinaisons légèrement différentes.

Dans le projet, les résultats montrent notamment qu'un modèle à quatre variables est très compétitif.

Les variables finalement retenues sont :

```text
X
DMC
DC
RH
```

où `RH` représente l'humidité relative.

---

# 23. Pourquoi choisir un modèle plus simple ?

Si deux modèles présentent des performances similaires, il est généralement préférable de choisir le modèle le plus simple.

Un modèle plus simple est :

* plus facile à comprendre ;
* plus facile à interpréter ;
* plus facile à maintenir ;
* souvent plus facile à expliquer.

Dans ce projet, le modèle à quatre variables est donc retenu :

```text
X
DMC
DC
RH
```

---

# 24. Régression linéaire

Le modèle principal utilisé est la **régression linéaire**.

```python
model = LinearRegression()

model.fit(
    X_train_selected,
    y_train
)
```

Le modèle cherche une relation de la forme :

$$
y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n
$$

Dans ce projet, les coefficients permettent d'estimer l'influence des variables sélectionnées sur la superficie brûlée.

---

# 25. Prédiction

Après l'entraînement du modèle, les prédictions sont effectuées sur les données de test :

```python
y_pred = model.predict(X_test_selected)
```

Les valeurs prédites peuvent ensuite être comparées aux valeurs réelles.

---

# 26. Mesure de performance : RMSE

La métrique principale utilisée est la **RMSE** (*Root Mean Squared Error*).

Elle est définie par :

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

où :

* \(y_i\) représente la valeur réelle ;
* \(\hat{y}_i\) représente la valeur prédite ;
* \(n\) représente le nombre d'observations.

Plus le RMSE est faible, meilleures sont les prédictions.

Cependant, un RMSE isolé ne permet pas de déterminer si un modèle est réellement performant. Il faut le comparer à une référence.

---

# 27. Régularisation avec Ridge et Lasso

Une autre étape d'optimisation consiste à utiliser la **régularisation**.

Deux méthodes sont étudiées :

* Ridge ;
* Lasso.

Elles ajoutent une pénalité aux coefficients du modèle.

L'objectif est notamment de limiter les coefficients excessifs et de réduire certains problèmes liés à la complexité du modèle.

---

# 28. Ridge Regression

La régression Ridge ajoute une pénalité basée sur le carré des coefficients.

Le paramètre principal est :

```text
alpha
```

Plus `alpha` est important, plus la pénalisation est forte.

Dans le projet, différentes valeurs de `alpha` sont testées, allant notamment de 1 à 10 000.

---

# 29. Lasso Regression

La régression Lasso utilise une autre forme de pénalisation.

Une propriété particulièrement intéressante de Lasso est qu'elle peut réduire certains coefficients jusqu'à zéro.

Une variable dont le coefficient devient nul peut alors être considérée comme ne participant plus au modèle.

Dans le projet, les variables :

```text
DMC
DC
RH
```

présentent des coefficients non nuls dans le modèle Lasso.

Cela rejoint en partie les résultats obtenus avec la sélection automatique des variables.

---

# 30. Comparaison des modèles

Trois modèles sont comparés :

1. Régression linéaire ;
2. Ridge ;
3. Lasso.

Pour chacun, on calcule le RMSE.

Les résultats sont très proches.

La régression linéaire obtient le meilleur résultat, mais avec une différence très faible par rapport à Ridge et Lasso.

Cela signifie que la régularisation n'apporte pas d'amélioration significative sur ce jeu de données.

---

# 31. Modèle de référence : Dummy Regressor

Pour savoir si le modèle de Machine Learning est réellement utile, il faut le comparer à un modèle extrêmement simple.

On utilise pour cela un **Dummy Regressor**.

Ce modèle ne cherche pas réellement une relation complexe entre les variables.

Il prédit simplement une valeur de référence, ici essentiellement basée sur la moyenne de la variable cible.

Le RMSE obtenu est d'environ :

```text
108,906
```

alors que le modèle de régression linéaire obtient environ :

```text
108,909
```

Les deux résultats sont pratiquement identiques.

---

# 32. Interprétation du résultat

Cette comparaison est particulièrement importante.

Le modèle de Machine Learning n'est pratiquement pas meilleur que le modèle qui se contente de prédire la moyenne.

Cela signifie que les variables disponibles dans ce jeu de données permettent très difficilement de prédire la superficie brûlée avec une simple régression linéaire.

L'optimisation du modèle n'a donc pas produit une amélioration spectaculaire.

Cela ne signifie pas que le travail d'optimisation est inutile.

Au contraire, la démarche permet de déterminer de manière systématique les limites du modèle et du jeu de données.

---

# 33. Coefficient de détermination R²

Une autre métrique utilisée est le **coefficient de détermination R²**.

Le R² mesure la capacité du modèle à expliquer la variabilité de la variable cible.

De manière générale :

* un R² proche de 1 indique une forte capacité explicative ;
* un R² proche de 0 indique une capacité explicative très faible ;
* un R² négatif peut également apparaître lorsqu'un modèle est moins performant qu'une référence appropriée.

Dans ce projet, le R² est pratiquement égal à zéro.

Cela confirme que le modèle explique très peu la variabilité de la superficie brûlée.

---

# 34. Conclusion du projet

Le principal résultat du projet est que le modèle construit n'est pas suffisamment performant pour prédire correctement la superficie des incendies.

Cependant, le projet permet de mettre en pratique une démarche complète d'optimisation d'un modèle de Machine Learning.

La démarche suivie est la suivante :

```text
Exploration des données
        ↓
Analyse de la variable cible
        ↓
Nettoyage des données
        ↓
Transformation des variables
        ↓
Encodage des variables catégorielles
        ↓
Séparation X / y
        ↓
Séparation entraînement / test
        ↓
Imputation des valeurs manquantes
        ↓
Analyse des valeurs aberrantes
        ↓
Standardisation
        ↓
Sélection des variables
        ↓
Régression linéaire
        ↓
Ridge / Lasso
        ↓
Comparaison des modèles
        ↓
RMSE + R² + modèle de référence
        ↓
Interprétation
```

Cette méthodologie peut être réutilisée dans de nombreux autres projets de Machine Learning.

---

# 35. Pourquoi l'optimisation n'a-t-elle pas amélioré le modèle ?

Il est important de comprendre qu'une bonne méthodologie ne garantit pas nécessairement un excellent score.

Plusieurs raisons peuvent expliquer les faibles performances :

* le jeu de données est relativement petit ;
* la variable cible est fortement asymétrique ;
* les relations entre les variables et `area` sont faibles ;
* les incendies présentent des comportements extrêmes ;
* les variables disponibles ne contiennent peut-être pas suffisamment d'information pour expliquer la superficie brûlée ;
* une régression linéaire peut être trop simple pour représenter certaines relations complexes.

Le jeu de données original décrit d'ailleurs cette tâche de régression comme difficile.

---

# 36. Pistes d'amélioration

Plusieurs améliorations peuvent être envisagées.

## 36.1 Utiliser des modèles non linéaires

Une régression linéaire suppose une relation essentiellement linéaire entre les variables et la cible.

On pourrait tester :

* Random Forest ;
* Gradient Boosting ;
* d'autres modèles non linéaires.

Ces modèles peuvent être capables de détecter des relations plus complexes.

---

## 36.2 Utiliser d'autres métriques

Le RMSE ne doit pas être la seule métrique utilisée.

On peut également calculer :

### MAE

$$
MAE =
\frac{1}{n}
\sum |y_i-\hat{y}_i|
$$

La MAE est plus directement interprétable et moins sensible aux grandes erreurs que la RMSE.

On peut également continuer à utiliser :

* R² ;
* RMSE ;
* MAE.

---

## 36.3 Examiner les grandes erreurs de prédiction

Il est intéressant d'étudier les observations pour lesquelles le modèle commet les erreurs les plus importantes.

On peut rechercher :

* les observations correspondant aux incendies ;
* les observations dont `area = 0` ;
* les conditions météorologiques particulières ;
* les valeurs extrêmes.

Cette analyse peut permettre d'identifier des groupes d'observations pour lesquels le modèle fonctionne particulièrement mal.

---

## 36.4 Construire un pipeline

Pour un projet destiné à être utilisé dans un environnement professionnel, il est préférable de regrouper les différentes étapes dans un **Pipeline**.

Le pipeline peut intégrer :

```text
Imputation
    ↓
Standardisation
    ↓
Sélection des variables
    ↓
Modèle
```

Cela permet d'obtenir un processus plus propre, reproductible et adapté à la production.

---

## 36.5 Optimiser les hyperparamètres

Une autre possibilité consiste à rechercher systématiquement les meilleurs hyperparamètres.

Par exemple :

```text
alpha pour Ridge
alpha pour Lasso
nombre d'arbres pour Random Forest
profondeur des arbres
paramètres du Gradient Boosting
```

Une recherche systématique peut être réalisée avec des techniques telles que la validation croisée et la recherche d'hyperparamètres.

---

## 36.6 Créer de nouvelles variables

L'**ingénierie des caractéristiques** (*feature engineering*) consiste à créer de nouvelles variables à partir des variables existantes.

On peut notamment rechercher des interactions entre variables.

Par exemple :

```text
température × humidité
température × vent
vent × humidité
```

Ces nouvelles variables peuvent parfois permettre au modèle de mieux représenter les relations complexes présentes dans les données.

---

# 37. Les principales notions à retenir

### Variable cible

La variable que le modèle cherche à prédire.

Ici :

```text
area
```

### Variables explicatives

Les variables utilisées pour effectuer la prédiction.

Exemples :

```text
X
Y
DMC
DC
RH
température
vent
```

### Imputation

Processus permettant de remplacer les valeurs manquantes par des valeurs estimées.

### KNN

Méthode utilisant les observations les plus proches pour effectuer certaines estimations.

### Outlier

Observation inhabituelle ou très éloignée des autres observations.

### Standardisation

Transformation permettant de mettre les variables sur une échelle comparable.

### Feature Selection

Sélection des variables les plus utiles pour le modèle.

### Régression linéaire

Modèle cherchant une relation linéaire entre les variables explicatives et la cible.

### Ridge

Régression linéaire avec une pénalisation des coefficients.

### Lasso

Régression régularisée pouvant réduire certains coefficients à zéro.

### RMSE

Mesure de l'importance moyenne des erreurs de prédiction, avec une pénalisation plus forte des grandes erreurs.

### R²

Mesure de la capacité du modèle à expliquer la variabilité de la variable cible.

### Baseline

Modèle simple servant de référence pour déterminer si un modèle plus complexe apporte réellement une amélioration.

---

# 38. Synthèse générale

L'optimisation d'un modèle de Machine Learning ne consiste pas simplement à choisir un algorithme et à calculer un score.

Elle repose sur une démarche structurée :

1. **Comprendre les données.**
2. **Comprendre la variable cible.**
3. **Nettoyer les données.**
4. **Transformer les variables.**
5. **Traiter les valeurs manquantes.**
6. **Séparer correctement les données d'entraînement et de test.**
7. **Analyser les valeurs aberrantes.**
8. **Standardiser les variables lorsque cela est nécessaire.**
9. **Sélectionner les variables pertinentes.**
10. **Construire plusieurs modèles.**
11. **Utiliser la régularisation.**
12. **Comparer les modèles avec plusieurs métriques.**
13. **Comparer les résultats à une baseline.**
14. **Analyser les erreurs.**
15. **Améliorer progressivement le pipeline.**

Dans ce projet, cette démarche montre surtout une chose essentielle en Data Science :

> **Un processus d'optimisation bien réalisé ne garantit pas qu'un modèle deviendra performant.**

Il permet cependant de comprendre **pourquoi** un modèle est limité et de déterminer quelles améliorations peuvent être envisagées.

Le résultat final est donc aussi important par ce qu'il révèle sur les limites des données que par le score obtenu par le modèle.
https://youtu.be/-ZYmnOY2VMk?si=xPZDSVKXXvRhNlh_
