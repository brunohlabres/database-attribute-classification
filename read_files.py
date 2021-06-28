import io, string, unicodedata, csv, re
import pandas as pd

def remove_accentuation(input_str):
	"""
	Returns a string with the accentuation stripped

		Parameters:
			input_str (str) : String to be transformed

		Returns:
			only_ascii (str): String stripped of accentuation
	"""
	nfkd_form = unicodedata.normalize('NFKD', input_str)
	only_ascii = nfkd_form.encode('ASCII', 'ignore')
	only_ascii = only_ascii.decode("utf-8")

	return only_ascii

def read_data_texts(file_texts):
	"""
	Returns a list of strings obtained from a file

		Parameters:

		Returns:
	"""
	print("Reading texts...")

	with open(file_texts, 'r') as file:
	    texts = file.readlines()

	texts = [remove_accentuation(t).lower().replace('\n','') for t in texts]

	print("Data contains {} samples of texts as input.\n".format(len(texts)))

	return texts

def read_data_names(file_names, atr_nomes, separator):
	print("Reading names...")

	data = pd.read_csv(file_names, sep=separator,quoting=csv.QUOTE_ALL, header=0, engine='python')

	for i in data:
		data[i].to_string()
	names = []

	for index, row in data.iterrows():
		s1 = str(row[atr_nomes[0]]).lower()
		s1 = remove_accentuation(s1)
		names.append(s1)

	nm = pd.Series(names).astype(str)

	print("Data contains {} samples of names as input.\n".format(len(nm)))

	return nm

def read_data_addr(file_addr, atr_addr, separator):
	print("Reading addresses...")

	data = pd.read_csv(file_addr, sep=separator,quoting=csv.QUOTE_ALL, header=0, engine='python')

	for i in data:
		data[i].to_string()

	addr = []

	for index, row in data.iterrows():
		s1 = str(row[atr_addr[0]]).lower()
		s1 = remove_accentuation(s1)
		addr.append(s1)

	address = pd.Series(addr).astype(str)

	print("Data contains {} samples of addresses as input.\n".format(len(address)))

	return address

def read_data(file_texts, file_names, file_addr):
	texts = read_data_texts(file_texts)
	names = read_data_names(file_names, ["NO_PROFISSIONAL"], separator=';')
	addr = read_data_addr(file_addr, ["NO_LOGRADOURO"],  separator=';')

	return texts, names, addr