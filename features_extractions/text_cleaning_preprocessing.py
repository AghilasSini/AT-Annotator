# Text Cleaning and Pre-processing
db = load_all_sentences();
print('chargement de {} vers dans la db'.format(len(db.keys())))


# # Premièrement, on récupère la fréquence totale de chaque mot sur tout le corpus d'artistes
# freq_totale = nltk.Counter()
# for k, v in corpora.iteritems():
#     freq_totale += freq[k]

# # Deuxièmement on décide manière un peu arbitraire du nombre de mots les plus fréquents à supprimer. On pourrait afficher un graphe d'évolution du nombre de mots pour se rendre compte et avoir une meilleure heuristique. 
# most_freq = zip(*freq2.most_common(100))[0]

# # On créé notre set de stopwords final qui cumule ainsi les 100 mots les plus fréquents du corpus ainsi que l'ensemble de stopwords par défaut présent dans la librairie NLTK
# sw = set()
# sw.update(stopwords)
# sw.update(tuple(nltk.corpus.stopwords.words('french')))



# # Tokenization
# # Stop words
# # Capitalization
# # Slangs and Abbreviations
# # Noise Removal
# # Spelling Correction
# # Stemming
# # Lemmatization


