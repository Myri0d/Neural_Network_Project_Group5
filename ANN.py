import pandas as pd
import os
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot
from numpy import array

def loadDoc(filename):
	file = open(filename, 'r', encoding="ISO-8859-1")
	text = file.read()
	file.close()
	return text


file = 'Database_design.txt'
#data = pd.read_csv('Consolidated_data.txt', sep=" ", header = None,lineterminator='\n')
#print(data)
data = defaultdict(list)
filehandle = open(file, 'r', encoding = "utf-8")
vocab = []
firstLine = True
def cleanDoc(document):
        
    document.replace('training_data.append({','')
    document = document.replace("sentence:",'')
    document = document.replace('class:',"")
    document = document.replace(')','')
    document = document.replace("{","").replace("}","")
    token = document.split()
#    print(token)
#    token = token.split('append(' )[1]
#    token = token.replace("\n","").replace('"',"")
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in token]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens
    
#    if(firstLine == True):
#        
#        vocab = Counter(s.split())
#        firstLine = False
#            
#    else:
#        vocab.update(s.split())
#    }
            
#    data[s.split(",")[0]].append(s.split(",")[1])
    
def addDocToVocab(filename, vocab):
	document = loadDoc(filename)
	tokens = cleanDoc(document)
	vocab.update(tokens)
    
    
#print(data)   
#print(counter)
# close the pointer to that file
filehandle.close()


#def save_list(lines, filename):
#	data = '\n'.join(lines)
#	file = open(filename, 'w')
#	file.write(data)
#	file.close()
    
def processDocs(directory, vocab, is_train):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('training'):
			continue
		if not is_train and not filename.startswith('training'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = docToLine(path, vocab)
		# add to list
		lines.append(line)
	return lines
        
def saveList(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
def docToLine(filename, vocab):
	# load the doc
	doc = loadDoc(filename)
	# clean doc
	tokens = cleanDoc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)


vocab_filename = 'set1.txt'
vocab = loadDoc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

sets_path = os.path.join(os.getcwd(),'sets')
positive_lines = processDocs(sets_path, vocab, True)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = positive_lines
tokenizer.fit_on_texts(docs)

# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')

ytrain = array([0 for _ in range(5)] + [1 for _ in range(5)])
 
# load all test reviews
positive_lines = processDocs(os.getcwd(), vocab, False)
docs = positive_lines

# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(1)])
 
n_words = Xtest.shape[1]
# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)

#Creates a counter of each word in the file and adds it to the total vocab
#processDocs('Databse_Design.txt', vocab)
#Example: processDocs('Access'.txt,vocab)
#print(vocab.most_common(50))
 
# save tokens to a vocabulary file
#saveList(tokens, 'vocab.txt')
#
#data = pd.DataFrame(data, index = ["class"])
#data = data.to_dict()
##print(data)
#data = pd.DataFrame([data])
path = os.path.join(os.getcwd(),'Data')
#os.makedirs(path)
#data.to_csv('CleanedData.csv',index =[1] )



# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest



# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 30
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(50, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=50, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores


def predictSentiment(review, vocab, tokenizer, model):
	# clean
	tokens = cleanDoc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])

#text = 'A covered entity may obtain consent of the individual to use or disclose protected health information to carry out treatment, payment, or health care operations'
text = 'In each row, the descriptive name is a link to the URL containing patient specific instructions'    
print(predictSentiment(text, vocab, tokenizer, model))