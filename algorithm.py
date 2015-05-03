import numpy as np

#X: Movies' feature vector (num_movies, num_features), each row is one movie's feature vector
#Y: User's rating for each movie: (num_movies, 1)
#Theta: User's preference vector: (1, num_features)


def getErrorFun(Theta,X,R,Y,alpha):
    InnerProduct=np.dot(X,Theta.transpose()[1:,:])+Theta[0,0]
    Error=np.multiply(R,np.square(np.subtract(InnerProduct,Y)))
    ErrorSum=np.sum(Error)/2
    ErrorReg=alpha/2*np.sum(np.multiply(R,np.square(Theta)))
    ErrorTotal=np.add(ErrorSum,ErrorReg)
    return ErrorTotal

def Iterate(Theta,X,R,Y,alpha,LearnRate):
    minus=np.subtract(np.dot(X,Theta.transpose()[1:,:]),Y)
    multi=np.outer(minus,Theta.transpose()[1:,:])
    regu=alpha*np.outer(np.ones((len(X),1)),Theta.transpose()[1:,:])
    total=np.multiply(R,np.add(multi,regu))
    inc1=-LearnRate*total.sum(0)
    total0=np.multiply(minus*Theta[0,0],R)
    inc0=-LearnRate*total0.sum()*np.ones((1,1))[0]
    inc=np.concatenate((inc0,inc1),1)
    return inc

def runIterate(Theta_init,X,R,Y,Reg,LearnRate,IterationTimes):
    Theta_list=[]
    Theta_list.append(Theta_init)
    Theta_new=Theta_init
    Error_list=[]
    Error_init=getErrorFun(Theta_new,X,R,Y,Reg)
    Error_list.append(Error_init)
    MovieScore=np.dot(X,Theta_new.transpose()[1:,:])+Theta_new[0,0]
    #print 'Initial Preference='+str(Theta_init)
    #print 'Initial Score='+str(MovieScore)
    #print 'Initial Error='+str(Error_init)
    for i in range(0,IterationTimes):
        #print 'Iteration '+str(i+1)
        inc=Iterate(Theta_new,X,R,Y,Reg,LearnRate)
        Theta_new=np.add(Theta_new,inc)
        Theta_list.append(Theta_new)
        MovieScore=np.dot(X,Theta_new.transpose()[1:,:])+Theta_new[0,0]
        Error=getErrorFun(Theta_new,X,R,Y,Reg)
        if i>4:
            if abs(Error-Error_list[-1]) < 0.00001:
                #Error>Error_list[-1] > Error_list[-2] or
                break
        Error_list.append(Error)
        #print 'New Preference='+str(Theta_new)
        #print 'New Score='+str(MovieScore.transpose())
        #print 'New Error='+str(Error)
        #print '   '
    #print ' '
    #print Error_list
    return (Theta_list,Error_list)
