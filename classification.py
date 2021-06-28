import io, string, itertools
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def init_sets(texts, names, addr):
	print("Extracting training, validation and test sets...\n")
	data_x = np.concatenate((texts, names, addr), axis=None)
	data_y = np.concatenate((np.full(len(texts), 1),np.full(len(names), 2),np.full(len(addr), 3)), axis=None)


	# Divides the training, validation and test sets in 80% 20%
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1, stratify=data_y)

	print("Training set size: {}".format(len(x_train)))
	print("Test set size: {}".format(len(x_test)))

	return x_train, y_train, x_test, y_test

def save_cm(array,file):
	plt.ylabel('Real')
	plt.xlabel('Classificado')

	target_names = ["Textos", "Nomes", "Enderecos"]
	sns.heatmap(array.astype('float')/array.sum(axis=1)[:, np.newaxis], annot=True, 
            fmt='.2%', vmin=0, vmax=1, cmap='Blues', xticklabels=target_names,yticklabels=target_names)

	plt.savefig(file)
	plt.close()

def get_digram_list():
	chars = []
	permut = []

	# guarda o alfabeto
	alphabet = string.ascii_lowercase

	for c in alphabet:
		chars.append(c)

	permut_list = itertools.product(chars, repeat=2)

	for l in permut_list:
		permut.append(''.join(l))

	return permut

def get_ordered_tfidf_digrams(data_x):
	vocab = get_digram_list()

	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='char_wb', ngram_range=(2,2), vocabulary=vocab)
	#vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='char_wb', ngram_range=(2,2), max_features=num_features)

	feature_array = np.array(vectorizer.get_feature_names())
	tfidf_mean = vectorizer.fit_transform(data_x).toarray().mean(axis=0)#.flatten()
	feature_mean = feature_array[np.argsort(tfidf_mean)][::-1]

	top_n = feature_mean[:len(vocab)]

	print("Ordered list of {} digrams with TF-IDF feature extraction:".format(len(vocab)))
	print(top_n)

	return top_n

def get_vectorizer_features(num_features, all_digrams):
	#np.asarray()
	dig = all_digrams[:num_features]
	#print(list[:num_features])
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, analyzer='char_wb', ngram_range=(2,2), vocabulary=dig)

	return vectorizer


def classify(x_train, y_train, x_test, y_test, num_features, vectorizer):

	x_train_f = vectorizer.fit_transform(x_train)
	x_test_f = vectorizer.transform(x_test) 

 	#---- NN -----
	# train
	clf_nn = MLPClassifier(random_state=1)

	clf_nn.fit(x_train_f, y_train)

	# cross validation
	cv = StratifiedKFold(5)
	scores_cv_nn = cross_val_score(clf_nn, x_train_f, y_train, cv=cv)
	print("Cross-validation: NN with {} features scored: {}".format(num_features, scores_cv_nn.mean()))
	print("Cross-validation: NN with {} features std: {}".format(num_features, scores_cv_nn.std()))

	# test
	y_pred_nn = clf_nn.predict(x_test_f)
	conf_mat_nn = confusion_matrix(y_test, y_pred_nn)
	save_cm(conf_mat_nn, "results/cm_nn_"+str(num_features)+".png")

	scores_nn = clf_nn.score(x_test_f, y_test)
	print(scores_nn)
	print("NN with {} features test score: {}".format(num_features, scores_nn))


	#---- SVM ----
	# train
	clf_svc = svm.LinearSVC(random_state=1)

	clf_svc.fit(x_train_f, y_train)

	# cross validation
	cv = StratifiedKFold(5)
	scores_cv_svc = cross_val_score(clf_svc, x_train_f, y_train, cv=cv)
	print("Cross-validation: SVC with {} features scored: {}".format(num_features, scores_cv_svc.mean()))
	print("Cross-validation: SVC with {} features std: {}".format(num_features, scores_cv_svc.std()))

	# test
	y_pred_svc = clf_svc.predict(x_test_f)
	conf_mat_svc = confusion_matrix(y_test, y_pred_svc)
	save_cm(conf_mat_svc, "results/cm_svc_"+str(num_features)+".png")


	scores_svc = clf_svc.score(x_test_f, y_test)
	print("SVC with {} features test score: {}".format(num_features, scores_svc))

	#------ RF -------
	# train
	clf_rf = RandomForestClassifier(random_state=1)

	clf_rf.fit(x_train_f, y_train)

	# cross validation
	cv = StratifiedKFold(5)
	scores_cv_rf = cross_val_score(clf_rf, x_train_f, y_train, cv=cv)
	print("Cross-validation: RF with {} features scored: {}".format(num_features, scores_cv_rf.mean()))
	print("Cross-validation: RF with {} features std: {}".format(num_features, scores_cv_rf.std()))


	# test
	y_pred_rf = clf_rf.predict(x_test_f)
	conf_mat_rf = confusion_matrix(y_test, y_pred_rf)
	save_cm(conf_mat_rf, "results/cm_rf_"+str(num_features)+".png")


	scores_rf = clf_rf.score(x_test_f, y_test)
	print("RF with {} features test score: {}".format(num_features, scores_rf))

	return scores_nn
