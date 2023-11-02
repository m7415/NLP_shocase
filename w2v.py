from gensim.models import KeyedVectors
import numpy as np
import pandas as pd

# fonction de vectorisation d'un texte
# wtv est optionnel. Si précisé, ce doit être un modèle word2vec 
#                    et model_vec_size doit être la taille des vecteurs de word2vec
# s'il n'est pas précisé, charge un modèle par défaut (c'est long)
def wtv_vectorizer(text,wtv : KeyedVectors=None, model_vec_size=None):
    if wtv == None or model_vec_size==None:
        print("no model given, defaulting")
        wtv : KeyedVectors = KeyedVectors.load_word2vec_format(
            './models/GoogleNews-vectors-negative300.bin', 
            binary=True,
            unicode_errors="ignore"
        )
        model_vec_size = 300

    # obtenir les vecteurs de chaque mot dans le texte, et faire la moyenne
    keys = [word for word in text.split() if word in wtv.key_to_index ]
    if len(keys) == 0: # s'il n'y a aucun mot reconnu (peu probable, mais il faut prendre en compte)
        return np.zeros( (model_vec_size) ) # on renvois un vecteur de 0
    mean = wtv.get_mean_vector( keys = keys )
    return mean

# fonction pour vectoriser tout un jeu de données
# les arguments sont les mêmes que pour wtv_vectorizer
def wtv_vectorizer_doc(doc,wtv:KeyedVectors=None,model_vec_size=None):
    if wtv == None or model_vec_size==None:
        print("no model given, defaulting")
        wtv : KeyedVectors = KeyedVectors.load_word2vec_format(
            './models/GoogleNews-vectors-negative300.bin', 
            binary=True,
            unicode_errors="ignore"
        )
        model_vec_size = 300

    # convertir en pd.Series
    if isinstance(doc, pd.DataFrame):
        doc = doc.squeeze()
    # convertir le dataframe en liste
    doc_list = doc.tolist()
    
    list_vectors = list(map(lambda txt: wtv_vectorizer(txt,wtv,model_vec_size), doc_list))
    stacked_vectors = np.vstack(list_vectors)
    return stacked_vectors

# cette fonction prend une liste (X_train ou X_test en fait)
# vectorise avec tout les modèles de word2vec chargés
# et renvois un dictionnaire de la forme :
# { nom_modèle_w2v : <données vectorisées> }
def vectorize(x, models): 
    combined_vectorized = {} # calcule une bonne fois pour toute les versions w2v de nos données d'entrainement
    for wtv, model_vec_size,name in models:
        print(f"Vectorisation avec w2v {name}")
        combined_vectorized[name] = wtv_vectorizer_doc(x,wtv,model_vec_size)
    return combined_vectorized
