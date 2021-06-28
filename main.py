import io
import pandas
from sources import *
from read_files import *
from classification import *


#if __debug__:
#	texts, names, addr = read_data(texts_debug, names_debug, addr_debug)
#else:
texts, names, addr = read_data(texts_src, names_src, addr_src)
# print (names)
# print (addr)
# print (texts)

x_train, y_train, x_test, y_test = init_sets(texts, names, addr)
all_digrams = get_ordered_tfidf_digrams(x_train)

qtd_features = [5,10,20,40,85,169,338,676]

for i in qtd_features:
	vectorizer = get_vectorizer_features(i, all_digrams)
	classify(x_train, y_train, x_test, y_test, i, vectorizer)
