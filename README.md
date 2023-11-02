# NLP_shocase
Some of the NLP notebook I worked on

## TAL (prédire le genre d'un film à partir de son synopsis)

Les données viennent d'une bdd d'alocine.

### Prétraitement

* NLTK, Spacy et TF-IDF
    * On a utilisé NLTK et Spacy pour faire un prétraitement des données (suppression des stopwords, ponctuation, lemmatisation, etc.)
    * On a utilisé TF-IDF pour représenter les données sous forme de vecteurs (voir [Modèles de classification](#modèles-de-classification))
    * On a utilisé les modèles de classification de SkLearn (voir [Modèles de classification](#modèles-de-classification))

* Word2Vec \
  Ce modèle n'est pas a priori utilisable pour représenter des phrases entières. Cependant, nous avons essayé une méthode pour s'en servir : 
  * on concatène le synopsis et le titre
  * on fait un prétraitement très simple (suppression des stopwords et de la ponctuation, mise en minuscules)
  * on fait la moyenne des vecteurs représentant chaques mots dans le texte
  
  Cette manière de faire nous fait bien sûr perdre beaucoup du sens du film, mais en passant cette représentation dans des modèles de classification (voir [Modèles de classification](#modèles-de-classification)), nous obtenons des prédictions qui dépassent les 50% de réussite, ce qui place cette méthode parmis nos meilleurs résultats

  Le modèle de Word2Vec que nous avons utilisé vient de M. Jean-Philippe Fauconnier (voir [Bibliographie](#bibliographie--sources))


### Modèles de classification

Nous avons utilisé plusieurs modèles de classification, ainsi que des transformers. Nous n'avons pas testé les réseaux de neurones.

* modèles de SkLearn :
  * Logistic Regression
  * CART (Decision Tree Classifier)
  * SVM
  * Random Forest
  * Decision Tree
  * KNN (K Neighbor Classifier)
  * Multinomial Naive Bayes
  * Dummy (renvois tout le temps le genre le plus fréquent, aucun apprentissage)

### Bibliographie / Sources

* Modèles Word2Vec : Jean-Philippe Fauconnier ([lien](https://fauconnier.github.io/#data)) \
  spécifiquement, nous avons testé 3 modèles :
  | dimension des vecteurs | algorithme utilisé pour l'entraînement | corpus utilisé                                                   | lien direct                                                                                                                |
  | ---------------------- | -------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
  | 200                    | cbow                                   | [FrWac corpus](http://wacky.sslmit.unibo.it/doku.php?id=corpora) | [lien de téléchargement (2.7Gb)](https://embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_cbow_cut0.bin)    |
  | 1000                   | cbow                                   | [FrWiki dump](https://dumps.wikimedia.org/frwiki/)               | [lien de téléchargement (253Mb)](https://embeddings.net/embeddings/frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin) |
  | 1000                   | skip                                   | [FrWiki dump](https://dumps.wikimedia.org/frwiki/)               | [lien de téléchargement (253Mb)](https://embeddings.net/embeddings/frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100.bin) |
  
  C'est le premier (vecteurs de taille 200) que nous avons retenu après tests

## Tweet analysis

Les notebooks english_dataset et english_dataset_w2v présentent de l'analyse de sentiments sur des tweets en anglais. L'un se base sur tf-idf, et l'autre sur un plongement word2vec (w2v).

## Installation

### create a virtual environment with python 3.6.9
```
python3 -m venv venv
```

### activate the virtual environment (linux)
```
source venv/bin/activate
```

### activate the virtual environment (windows)
```
source venv/Scripts/activate
```

### install the requirements
```
pip install -r requirements.txt
```