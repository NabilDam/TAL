import numpy as np
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords


def load_corpus(filename):
   
    with open(filename,"r") as f:
        lines = f.readlines()
        
    #stemmer = SnowballStemmer("french", ignore_stopwords=True)        
    
    X = []
    y = []
    for line in lines:
        y.append(line.split()[0][-2])
        #X.append(' '.join([stemmer.stem(m) for m in line.split()[1:]]))
        X.append(' '.join(line.split()[1:]))
    resultat = (np.array(X),np.array(y))
    
    return resultat




def construction_dico(X,Xt,tf_idf=False):
    #token = r"\b[^\d\W]+\b/g"
    #stemmer = SnowballStemmer("french", ignore_stopwords=False)
    if tf_idf:
        
        vectorizer = CountVectorizer(stop_words=stopwords.words('french'),token_pattern=r'[a-zA-Z]+')
        Xvec = vectorizer.fit_transform(X)
        X_tf_idf = TfidfTransformer().fit_transform(Xvec)
        XvecT = vectorizer.transform(Xt)
        X_tf_idf_test = TfidfTransformer().transform(XvecT)
        return Xvec,XvecT,vectorizer.get_feature_names(),X_tf_idf,X_tf_idf_test
    
    else:
        vectorizer = CountVectorizer(stop_words=stopwords.words('french'),token_pattern=r'[a-zA-Z]+')
        Xvec = vectorizer.fit_transform(X)
        XvecT = vectorizer.transform(Xt)
        return Xvec,XvecT,vectorizer.get_feature_names()

def random_predict(Xt):
    file = open("predicte_random.txt","w")
    for i in Xt:
        prediction = np.random.choice(["C","M"])
        file.write(prediction+"\n")
    file.close()
    
def write_prediction(prediction,lissage=True):
    file = open("predicte_svm.txt","w")
    for p in prediction:
        if lissage:
            if p ==1:
                p = 'C'
            else:
                p='M'
                
        file.write(p+"\n")
    file.close()

def lissage_prediction(prediction):
    
    for i in range(5,len(prediction)-5):
        if(sum([ 1 for p in prediction[i-5:i+4] if p == 'C']))>(10-sum([ 1 for p in prediction[i-5:i+4] if p == 'C']))*2.85:
            prediction[i] = "C"
        else:
            prediction[i] = "M"
    
    for i in range(1,len(prediction)-1):
        if prediction[i]!=prediction[i-1] and prediction[i]!=prediction[i+1]:
            prediction[i] = "C" if prediction[i] == "M" else "M"
    
    return prediction


#Post processing 
        
coeff_M=2.85

def lissage(Pred):
    

    for i in range(5,len(Pred)-5):
        tmp=np.concatenate((Pred[i-5:i],Pred[i+1:i+6]),axis=None)
        nb_M=np.count_nonzero(tmp == 'M')
        
        nb_C=np.count_nonzero(tmp == 'C')
        if nb_M*2.85>nb_C :
            Pred[i]='M'
        else:
            Pred[i]='C'
    
    return Pred
        
def post_lissage(Pred):
    tmp = Pred.copy()
    for i in range(1,len(Pred)-1):
        if tmp[i-1]!=tmp[i] and tmp[i-1]==tmp[i+1]:
            Pred[i]=Pred[i-1]
    Pred[i]=tmp[-1]
    return Pred

if __name__ == "__main__":
    
    X,y = load_corpus("data/president/corpus.tache1.learn.utf8")
    Xt,yt = load_corpus("data/president/corpus.tache1.test.utf8")
    print(y)
    (Xvec,XvecT,vect) = construction_dico(X,Xt,False)
    # données ultra basiques, à remplacer par vos corpus vectorisés

    #random_predict(Xt)
    # SVM
    clf = svm.LinearSVC(class_weight='balanced')
    # Naive Bayes
    #clf = nb.MultinomialNB()x
    # regression logistique
    #clf = lin.LogisticRegression()
    
    # apprentissage
    clf.fit(Xvec, y)  
    #print(clf.predict([[2., 2.]])) # usage sur une nouvelle donnée
    predict =clf.predict(XvecT)
    predict1=post(list(predict))
    predict_f = post2(list(predict1))
    write_prediction(predict_f,lissage=True)