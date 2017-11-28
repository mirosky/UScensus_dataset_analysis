# UScensus dataset analysis for Dataiku interview

## Objectif

L'objectif de cette étude est, dans un premier temps d'investiguer statistiquement un jeux de données issu de recensement et, dans un second temps, de construire un modèle de prédiction qui nous permettra de classifier automatiquement si un individu a un revenu plus ou moins important que 50 000 dollars par an.

## Les données

### Description 
Les données fournies sont issues de la base intitulée "US Census". Il s'agit de données anonymisée recensées sur environs 300 000 personnes.

Nous avons à disposition :

- une base d’apprentissage `census_income_learn.csv`,
- une base de test `census_income_test.csv`,
- un fichier `census_income_metadata.txt` avec quelques explications sur les données.

Nous constatons que la taille des données d'entraînement (199 523 lignes) est trois fois plus importante que la taille du jeux de test (61 342 lignes  et non 99 762 comme précisé dans le fichier de métadonnées), cette répartition est donc plutôt équilibrée pour un apprentissage consistant et une évaluation cohérente (ratio 2/3 et 1/3).

### Détail des attributs

On appellera “attributs” l'intitulé de chaque colonne des fichiers d'entraînement et de test, il s’agit donc de l'appellation d’une variable descriptive. En étudiant minutieusement les noms associés à chaque variable recensée, on note un oubli dans le fichier de métadonnées : en effet l’information représentée par la réponse “both parents” dans les observations n’a pas d'intitulé correspondant. On l’ajoute donc à notre liste d’attributs juste après celui intitulé “family members under 18” et l’on choisit de le nommer “presence of family members" par déduction.

Nous avons donc 42 variables : 41 attributs observés pour chaque personne plus la variable que l’on va chercher à prédire : notre vérité terrain labellisée par “- 50000” ou “+50000” (qui sera tagué dans notre code par `GT` pour “Ground Truth”).

Parmis les variables, 34 sont dites nominales (valeurs textuelles) et les 7 autres sont dites continues (valeurs numériques). Il est important de bien comprendre la signification de chaque variable en se posant la question pour chacune : Est-ce que cet attribut est décisif pour mon problème, c'est à dire reflète t il les revenus d'une personne ou peut il l'influencer (aspect économique ou critère social important) ?

Nous pouvons d'ores et déjà faire un premier tri (on taguera par “important” les variables à garder qui nous paraissent judicieuses pour notre étude, et par “ignore” les variables que l’on décide de ne pas prendre en compte) : 

#### pour les variables continues : 

   - __age__ -> important.
   - __detailed industry recode__ code du type d'activité détaillé de l'entreprise et __detailed occupation recode__ code du type d’activité détaillé  de la personne 
(grâce au lien `https://www.bls.gov/cps/cenind2012.pdf). On ignore ces deux codes car on a l'information d'activité dans les variables __major occupation code__ et __major industry code__, plus facilement interprétables que des codes. -> ignore. 
   - __wage per hour__ salaire par heure -> important.
   - __capital gains__ profit des investissements -> important.
   - __capital losses__ perte des investissements -> important.
   - __dividends from stocks__ part des actions dans l'entreprise -> important.

#### pour les variables nominales : 

   - __class of worker__  -> important.
   - __major industry code__ domaine d’activité général. On a l'information plus précise grâce à __major occupation code__  -> ignore.
   - __major occupation code__ type de corps de métier -> important.
   - __education__ degré du diplôme obtenu -> important.
   - __enroll in edu inst last wk__ impliqué dans une institution scolaire la semaine dernière : incompréhension de ce champ -> ignore.
   - __marital stat__ -> important.
   - __race__ -> important.
   - __hispanic origin__ information très précise et déductible d'autre données comme par exemple __country of birth self__ -> ignore.
   - __sex__ -> important.
   - __member of a labor union__  prise de parti dans un syndicat -> ignore.
   - __reason for unemployment__ on ne prend en compte que les individus gagnant un salaire -> ignore.
   - __full or part time employment stat__ on se doute que les personnes qui gagnent plus ne sont pas à mi temps -> ignore.
   - __tax filer stat__ statut de la personne qui remplit les taxes/impôts -> ignore.
   - __region of previous residence__  redondant avec __state of previous residence__ : si on connait la région, on connait l'état -> ignore.
   - __state of previous residence__ de quel etat la personne vient -> important.
   - __detailed household and family stat__ composition de la famille -> ignore.
   - __detailed household summary in household__ rôle familial-> ignore.
   - __migration code-change in msa__ la personne vient t'elle d'un  regroupement de comtés / d’une aire urbaine ? -> important.
   - __migration code-change in reg__ le champs __migration code-change in msa__ est plus important -> ignore.
   - __migration code-move within reg__ information trop générale, plus important de se focaliser sur la MSA qui fournit un aspect économique , de plus on a déjà l’information de la localisation dans __state of previous residence__  -> ignore.
   - __live in this house 1 year ago__ -> ignore.
   - __migration prev res in sunbelt__ la personne vient t’elle de la partie sud des USA ? On a deja cette information grace à __state of previous residence__ -> ignore.
   - __num persons worked for employer__ nombre de personne dans l’entreprise -> ignore.
   - __family members under 18__ membre de la famille de moins de 18 ans > ignore.
   - __country of birth father__ -> ignore
   - __country of birth mother__ -> ignore
   - __country of birth self__ on privilégiera d'étudier la nationalité grâce à __citizenship__ plutot -> ignore.
   - __citizenship__ nationalité -> important.
   - __own business or self employed__ la personne a t’elle son propre business ou s'emploi t’elle ? -> important.
   - __fill inc questionnaire for veteran's admin__ statut de veteran -> ignore.
   - __veterans benefits__ avantages pas financiers -> ignore.
   - __weeks worked in year__ reflète la quantité de travail -> important.
   - __year__ à étudier en cas de crise économique sur une année -> important.
   - __presence of family members__ -> ignore.

Nous avons donc écarté certains attributs qui, sans analyse statistique, de premier abord, ne paraissent pas pertinent à prendre en compte par rapport à notre problématique.
On se concentre donc sur les 17 attributs :
 
```
age' 'class of worker' 'education' 'wage per hour' 'marital stat'
 'major occupation code' 'race' 'sex' 'capital gains' 'capital losses'
 'dividends from stocks' 'state of previous residence'
 'migration code-change in msa' 'citizenship'
 'own business or self employed' 'weeks worked in year' 'year
```

## Qualité des données

Il est important de vérifier la qualité des données, c'est à dire est-ce que les champs sont renseignés dans suffisamment d’observations afin d'avoir une étude consistante.
On note que l’on travaille essentiellement sur les données du fichier d’apprentissage  `census_income_learn.csv` afin de ne pas biaiser la création du modèle de prédiction qui sera appliqué à la toute fin sur les données de test  `census_income_test.csv`, non étudiées donc.
On comptabilise donc, pour chaque attribut son pourcentage de  “Not in universe” correspondant aux “missing value”. Voici une définition (“http://answers.popdata.org/”) plus précise :


_"Not in Universe" implies that the person was not a part of the population to which the question was directed.
The target population is called the question universe, and you can tell who was meant to be included in the universe in a variable's Universe Statement.
Example : the variable “pregnancy” for a male._


On décide de supprimer les attributs contenant moins de 40% de “bonnes données” c’est à dire, si il y a plus de 60% de “missing values” pour cette variable.

Les champs à exclure sont :

```
enroll in edu inst last wk, member of a labor union,
reason for unemployment, region of previous residence,
presence of family members,
fill inc questionnaire for veterans admin et  state of previous residence.
```

Ce ne sont pas des variables qui apporteront une très grandes plus-value pour l’analyse puisqu’elles sont renseignées pour peu de personnes (moins de 40%).  On peut donc amputer ces champs de nos données. On remarque que tous les champs avaient été exclus au préalable excepté __state of previous residence__, ce qui confirme que ces attributs étaient trop précis pour pouvoir être recensé pour tous les individus.

Nous en avons donc à présent 16 attributs pour notre audit statistique.

```
age' 'class of worker' 'education' 'wage per hour' 'marital stat'
 'major occupation code' 'race' 'sex' 'capital gains' 'capital losses'
 'dividends from stocks' 'migration code-change in msa' 'citizenship'
 'own business or self employed' 'weeks worked in year' 'year
```

## Etude statistique univariée

### Générale

Afin d’obtenir des statistiques de base sur chacun des attributs de manière indépendante les un entre les autres, on utilise la fonction  `describe` de panda de deux façons différentes : pour les données nominales et pour les données continues.

On observe que pour une variable continue, l'âge par exemple , la moyenne des individus est de 34 ans, l'écart type de 22 ans, le plus jeune âge de 0 et le plus vieux de 90 ans. On note qu’il y a 25% des gens au dessous de 15 ans, 50% des gens au dessous de 33 ans et 75% des gens au dessous de 50 ans.
Parmis les données continues, nous avons le descriptif de chaque classe (plus ou moins de 50K). On vois qu'il y  a 187141 personnes qui gagnent moins de 50K$ par an sur 199523 donc seulement 6,20% des individus. 

Pour une variable  nominale, le champ représentant le sexe des gens par exemple, on observe qu’il y a  plus de femmes dans les observations au nombre de 103 984 sur 199 523 (52%). Pour ces données dites nominales, il est intéressant d'examiner le champs “unique” qui représente le nombre d’observations différentes par variable. Ici le max d'occurrence se situent dans la catégorie “éducation” et est de 17, sur 199 523 observations cela n’est pas aberrant, en revanche il ne faudrait pas prendre en compte une variable qui a presque autant de réponse différentes que d’observations.

Aussi, on constate que pratiquement toutes les personnes ont la nationalité “Native-born in the United States” (176992 sur 199523 donc 89%) donc cette information n’est pas très discriminante pour notre problème.

De plus, grâce aux fonction `groupby` et `mean`, on peut mettre en exergue les données numériques en fonction de chacune des classes.

Parmis les données les plus différentes entre les deux classes, on lit que les personnes qui gagnent moins de 50K par an (VS celles qui gagne plus de 50K par an) :

- On environs 33 ans (VS 46 ans),
- on un salaire par heure moyen de 53 dollars (VS 81 dollars),
- on un capital gain moyen de 143 (VS 4830),
- on un capital loss moyen de 27 (VS 193),
- on un divident from stock moyen de 107 (VS 1553),
- on un migration codechange in msa moyen de 1736 (VS 1796),
- travaillent en moyenne 21 semaines par an (VS 48).

En revanche, parmis les données qui ont presque les mêmes moyennes et écart types dans les deux groupes on note : le __migration code-change in msa__, __year__ ainsi que __citizenship__. On les supprime donc de nos données vu que ce ne seront pas des features très discriminantes pour notre classifieur.

Nous avons donc à présent 13 attributs : 

```
age' 'class of worker' 'education' 'wage per hour' 'marital stat'
 'major occupation code' 'race' 'sex' 'capital gains' 'capital losses'
 'dividends from stocks' 'own business or self employed'
 'weeks worked in year'
```

### Par variable

Pour avoir une meilleure idée du comportement de chaque variable, on construit une visualisation des différentes distributions de chacune des variables (principalement grâce à la fonction `countplot`). Voir le diagramme générale dans images/distributions.png.
Dessus, on peut lire les valeurs extrêmes (max, min) et surtout les fréquences pour chaque attributs. Par exemple : 

- il y a 2 pics d'âge autour de 8 et 35 ans,
- la plupart des gens travaillent dans le privée mais ce champs n’est pas beaucoup renseigné,
- la plupart des gens ont une éducation “children” ou “high school graduate”,
- le revenu par heure est difficilement lisible car concentré principalement en une valeur, pareil pour le capital gains, le capital losses et le dividents from stocks,
- la majeure partie des individus n’ont jamais été mariés, et presque autant sont marié(e)s civilement avec un(e) époux(se) présent(e),
- les corps de métiers principaux sont les domaines de la vente, l’administration, le management mais ce champs n’est pas beaucoup renseigné,
- pratiquement toutes les personnes sont blanches,
- il y a un peu plus de femmes que d’hommes recensées,
- quelques personnes ont leur propre business, un peu plus s’emploient eux même mais ce champs n’est pas beaucoup renseigné, 
- il ya 2 pics pour le nombres de semaines travaillées par ans, un autour de 0 et un autour de 50 ce qui doit représenter les personnes en poste ou non.

### Par variable et par classe 

Une chose intéressante aussi à visualiser est la distributions de chacune des variables en fonction de chaque classe. Voir le diagramme générale dans images/distributions_per_class.png. On notera que l’on a regroupé certaines variables dans des bins pour plus de lisibilité.

On y lit par exemple que les gens qui gagnent plus de 50K par an (VS ce qui gagnent moins de 50K) sont majoritairement :

- dans la tranche d’âge [36-45] ans ( VS [0-9]),
- travailleur dans le privé (VS privé aussi),
- titulaires d’un bachelor ou un master (VS children et high school),
- marié(e)s avec un(e) époux(se) présent(e) (VS n’ont jamais été mariés),
- travailleur dans le management (VS administration),
- travaillent 52 semaines par an (VS 0).

Certaines données liées au gain losses, salaire par heure et dividends from stocks ne sont pas lisibles grâce à cette représentation et d’autres pas très intéressantes car quasi-identiques dans les deux classes.

### Entre les variables

Une chose importante aussi en analyse de données est de regarder quelles sont les variables qui sont très corrélées, si c’est le cas elles encodent donc presque la même information et sont redondantes.  On peut voir l’interaction de variables entre elles grâce à la la matrice de corrélations, symétrique et diagonale. Voir le diagramme : ./images/correlation_matrix.png.

On notera qu’avant de pouvoir visualiser cette matrice, il faut normer la représentation des variables. En effet,  les features finales doivent être toutes continues (et non nominales). On pourrait les transformer manuellement par exemple en changeant l’attribut education en définissant une échelle numérique de niveaux mais certains groupes ne peuvent pas être ordonnés comme la race ou le statut marital par exemple. Ainsi, on applique une technique appelée “label encoding” pour constituer notre jeux de features final qui sera injecté dans notre classifieur. Il s’agit simplement de convertir chaque valeur d’une colonne en un nombre.

J’ai souhaité tester l’approche “one hot encoding” qui consiste à convertir chaque catégorie en une nouvelle colonne et assigner des 1 ou 0 (True or False) correspondant, mais malheureusement la fonction de panda `get_dummies` me créer des erreurs de mémoire que je n’ai pas réussi à résoudre.

Ainsi, sur la matrice de corrélations, on visualise clairement  un fort lien entres certaines variables comme par exemple  entre le nombre de semaines travaillés par an et la classe du travailleur. A contrario, il n’y a par exemple une faible corrélation entre le nombre de semaines travaillés par an et le statut marital de la personne. Ces informations pourraient nous permettent de créer manuellement de nouvelles caractéristiques en combinant et croisant plusieurs attributs ensembles.

# Construction du classifieur

Nous allons donc créer un classifieur qui répondra à la question : étant donné les données de recensement d’une personne donnée, celle ci gagne t elle plus ou moins que 50K par an ?

##  Types de modèles

Le package `scikit learn` de python fournit une grande variété de modèle de classifieurs standards comme la régression logistique, les KNN, le SVM, des réseaux de neurones, les random forest, etc. Dans le cadre de cette étude, je testerais la régression logistique puisque notre problématique est de relier la un événement à une combinaison linéaire de variables explicatives. En effet, on cherche bien dans notre cas à mesurer la survenu d’un événement aux facteurs susceptibles de l’influencer. On choisi un modèle linéaire pour la rapidité d'exécution lors de l’étape du “fitting”. On testera aussi les KNN (K-plus proches voisins) et le SVM.

Afin que les algorithmes donnent des résultats probants, les variables doivent être indépendantes les unes des autres au maximum, ce qui a été vérifié grâce à notre matrice de corrélation précédemment qui a confirmé que nous avons bien gardé des variables significatives. Ainsi, notre modèle ne sera pas trop “multi colinéaire”.  De plus, nous devons avoir suffisamment d’exemples.
Nous choisissons de découper notre base d'entraînement, en 2/3 pour l’apprentissage et 1/3 pour la validation afin de pouvoir évaluer notre algorithme et ne surtout pas toucher au fichier test fournis dans les données.
On fera bien évidemment attention à supprimer les potentielles valeurs NaN grâce à la fonction `dropna`. 
 
## Evaluation

Une fois l’étape de “fitting” processée, nous appliquons notre modèle entraîné sur les données de validation, on compare donc les probabilités prédites pour chaque observation avec la vérité terrain des données de validation. Ainsi, nous avons un score de précision. 

Sans aucun tuning des paramètres, nous obtenons pour la régression logistique une précision de 94% et un rappel de 95% en 3 secondes environs.
Le détail par classe est donné grâce à la fonction `classification_report`.
On teste aussi la technique de cross validation avec 10 folds, les scores sont à peu près les mêmes.

De même pour les KNN “non tunés”, on obtient une précision de 93% et un rappel de 94% en 144 secondes environs.
De même, la cross validation donne à peu près les mêmes résultats.
Les tests ont été réalisés sur un portable Asus X302L, CPU Intel Core i5.

Par ailleurs, le SVM n’a jamais convergé, la frontière doit être trop complexe à trouver il y aura donc un risque d’overfitting sur les données d'entraînement, on le supprime des test. J’aurais souhaité construire une grille de recherche optimisée pour choisir les meilleurs paramètres de chaque modèle mais malheureusement, ma machine ne supporte pas le calcul parallèle.  Aussi, il aurait été judicieux de tracer les courbes ROC pour montrer les taux de vrais positifs et faux positifs  chaque seuils sur les probabilités en sortie.

## Test 

On décide donc de choisir le modèle de régression logistique linéaire donnant de bons scores en un temps de calcul rapide. On ré-entraîne le model sur toute la base d’apprentissage ( `census_income_learn.csv`) ( et non juste la sous partie comme précédemment). Et, on applique le nouveau modèle sur les vrais données de test (`census_income_test.csv`).

Le F-score est de 49%, la précision est de 94% mais le rappel seulement de 38%. Ceci veut dire que l’on rate des vrais positifs donc on ne détecte pas toutes les personnes qui gagnent plus de 50K par an. En revanche, lorsque quand on en prédit une, c'est quasiment toujours bon, il n’y a presque pas de faux positif. Le système est donc peu bruité, il donne des résultats fiables même s’ils sont partiels. Dans tout problème de classification, il faut trouver le juste milieu entre la précision et le rappel qui convient le plus à notre problématique. 
 
# Réflexions et pistes futures 

Nous avons pu observé que les facteurs qui jouent sur le fait de gagner plus de 50K par an est le fait d'être un homme blanc, d’environ 46 ans, marié, citoyen américain travaillant 52 semaines par an. 
 Les étapes de cette étude ont vraiment été de comprendre les données, sélectionner les pertinentes, les nettoyer, étudier leur qualité et leurs statistiques, les encoder correctement, puis bien sûr construire des classifieurs.
Aussi, par rapport à notre problématique, on aurait pu sélectionner une sous base des personnes ayant au moins 16 ans, faire du “feature crossing” comme exposé précédemment c’est à dire créer de nouvelles variables intéressantes, et faire du “feature selection and reduction” grâce à des algorithmes de PCA (pour corréler des variables et donc réduire le nombre de features ) ou bien SVD afin de n’avoir que les features les plus importants et réduire la dimensionnalité. Enfin, il aurait été intéressant aussi de tester plusieurs types d’encodage (hot) pour les variables nominales.

 



 

 
 
