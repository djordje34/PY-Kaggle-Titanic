import numpy as np
#from numpy.core.fromnumeric import reshape
import pandas as pd                        #iz razloga sto nemamo postojecu util biblioteku koristio sam pandas za R/W dokumenata 

def main(train_path, valid_path, save_path):                            #ovi pathovi su inspirisani pathovima sa domacih zadataka
    #train=pd.read_csv('train.csv')                                         #moguce je zameniti stringove dole tipa "test.csv" sa valid_path
    #train['Age'].fillna((train['Age'].mean()), inplace=True)                #fillujemo srednjim vrednostima godina  
                                                                                        #OVO TRAIN JE NEPOTREBNO NAPISANO JE RADI PROVERE METODE FILLNA!!!!
                            #popunjava null cellove!
    clist=['Pclass','Sex','Age','SibSp','Parch','Fare']                 #lista headera koje sam koristio zarad procene prezivelih
    x_data=pd.read_csv('train.csv',usecols=clist)                       #['Pclass']['Sex','Sex','Age','Sibsp','Parch','Fare'];
    x_data['Age'].fillna((x_data['Age'].mean()), inplace=True)              #isto kao gore, x_data sam koristio
    survived=['Survived']                                                       #linija ispod cita niz argumenata po kojima izdvaja kolone i kopira u novi niz, komanda radi samo sa nizovima
    y_data=pd.read_csv('train.csv',usecols=survived)                                
    x_train=x_data.to_numpy()                                                       #kao ispod
    y_train=y_data.to_numpy()                                               #castujemo u numpy array kako bi mogli da manipulisemo celijama matrice
    for i in range(len(x_train)):
        for j in range(len(x_train[0])):                                        #ako je musko 0 ako zenko 1 jer ne znam kako da implementujem da log regresija radi sa stringovima
            if(x_train[i][j]=="female"):
                x_train[i][j]=1
            elif(x_train[i][j]=="male"):
                x_train[i][j]=0
    x_train=x_train.astype(float)                                       #castujemo u float da bi mogli da radimo operacije deljenja i mnozenja bez gubitaka
    y_train=y_train.astype(float)        
    y_train=y_train.flatten()                                           #flatten da castuje y(len(x),1) u 1-D niz y(len(x)), zbog problema sa .dot operaciju sa hipotezom
    
    #np.savetxt('info.txt', x_train, fmt='%f')                               #nepotreban text doc!!!!!

    model=LogisticRegression()
    model.fit(x_train,y_train)
    x_test=pd.read_csv('test.csv',usecols=clist)    
    x_test['Age'].fillna((x_data['Age'].mean()), inplace=True)                  #fillujemo kao gore za test skup
    x_test=x_test.to_numpy()                                                        #isto...

    for i in range(len(x_test)):
        for j in range(len(x_test[0])):
            if(x_test[i][j]=="female"):
                x_test[i][j]=1
            elif(x_test[i][j]=="male"):
                x_test[i][j]=0
    x_test=x_test.astype(float)
    ids=['PassengerId']
    id_test=pd.read_csv('test.csv',usecols=ids) 
    id_test=id_test.to_numpy()
    id_test=id_test.flatten()                               #isto kao za y_train gore
    y_predict = model.predict(x_test)
    y_predict=np.round(y_predict)               #ako predict>05 preziveo/la ako manje nije
    y_predict=y_predict.astype(int) 
    id_test=id_test.astype(int)                     #castujemo u int
    np.savetxt('Submission.csv',np.c_[id_test,y_predict],fmt='%d',header="PassengerId,Survived",comments='') #spajamo pomocu np.c_ idtest i ypredict u formatu int 
class LogisticRegression():
    
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,theta_0=None, verbose=True): #izmenjeni logreg sa prvog domaceg, logreg sa Hesijanom najbolji iz razloga sto daje cist decision-boundary za vise argumenata
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
    def get_eps(self,h_x,y):                #vraca trenutnu pretpostavku                                                                                      
        return np.sum(y * (h_x)+ (1 - y) * (1 - h_x))   
    def fit(self, x, y):
        
        n=len(x)
        #print(y)
        m=len(x[0])
        hip_x=np.zeros((n,1))
        self.theta = np.zeros(m)            #vraca niz nula duzine m, koji cemo u Njutnovoj metodi da preradimo
        current_iter=0
        while current_iter<self.max_iter:                                 #Njutnova metoda
            hip_x = 1 / (1 + np.exp(x.dot(-(self.theta))))          #racunamo hipotezu
            H = np.dot((x.T * hip_x * (1 - hip_x)),(x)) / n              #Hessian matrica
            H_inv=np.linalg.inv(H);                                         #H^{-1}
            J_theta = x.T.dot((hip_x - y)) / n                           #J_{\theta}
            #print(y)
            #print(hip_x.shape,y.shape,(hip_x-y).shape,type(hip_x),type(y),J_theta.shape,self.theta.shape)
            self.theta -= self.step_size*(H_inv.dot(J_theta))                #update theta
            hip_x_n=1 / (1 + np.exp(x.dot(-(self.theta))))
            old_eps=self.get_eps(hip_x,y)                       #racunamo staru preciznost
            curr_eps=self.get_eps(hip_x_n,y)                       #novu preciznost
            if abs(curr_eps-old_eps) <= self.eps:       #proveravamo razliku eps   
                break
            current_iter+=1
            
            
    def predict(self, x):
        hip_x=1 / (1 + np.exp(x.dot(-(self.theta))))             #decision
        return hip_x  
        
        
        
        
if __name__ == '__main__':
    main(train_path='train.csv',
         valid_path='test.csv',
         save_path='info.txt')