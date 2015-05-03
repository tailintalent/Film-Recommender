import numpy as np

#Initialize vectors and parameters
Y=np.matrix('4;2;3')    #Rating, imported from website
R=np.matrix('1;1;1')   #vector indicating whether a user rates a movie
UserPrefer_init=np.matrix('0 0.2 0.1 0.4 0.3')  #the first element is theta_0
Reg=0.01   #Regulization
LearnRate=0.01
IterationTimes=100

#import movie feature vector X from JSON type list
MovieList=[{'X':[1,2,5,7]},{'X':[4,2,5,3]},{'X':[3,5,3,2]}]
num_movies=len(MovieList)
num_features=len(MovieList[0]['X'])
X=np.zeros((num_movies,num_features))   #Movie feature vector
for i in range(num_movies):
    X[i]=MovieList[i]['X']

print X

def getErrorFun(UserPrefer,X,R,Y,alpha):
    InnerProduct=np.dot(X,UserPrefer.transpose()[1:,:])+UserPrefer[0,0]
    Error=np.multiply(R,np.square(np.subtract(InnerProduct,Y)))
    ErrorSum=np.sum(Error)/2
    ErrorReg=alpha/2*np.sum(np.multiply(R,np.square(UserPrefer)))
    ErrorTotal=np.add(ErrorSum,ErrorReg)
    return ErrorTotal

def Iterate(UserPrefer,X,R,Y,alpha,LearnRate):
    minus=np.subtract(np.dot(X,UserPrefer.transpose()[1:,:]),Y)
    #print 'tentative rating='+str(np.dot(X,UserPrefer.transpose()[1:,:]))
    #print 'actual rating='+str(Y)
    #print 'difference='+str(minus)
    multi=np.outer(minus,UserPrefer.transpose()[1:,:])
    #print multi
    regu=alpha*np.outer(np.ones((len(X),1)),UserPrefer.transpose()[1:,:])
    #print regu
    total=np.multiply(R,np.add(multi,regu))
    #print total
    inc1=-LearnRate*total.sum(0)
    total0=np.multiply(minus*UserPrefer[0,0],R)
    inc0=-LearnRate*total0.sum()*np.ones((1,1))
    inc=np.concatenate((inc0,inc1),1)
    #print 'UserPrefer='+str(UserPrefer)
    #print 'incremental='+str(inc)
    return inc

def runIterate(UserPrefer,X,R,Y,alpha,LearnRate,IterationTimes):
    Userprefer_list=[]
    Userprefer_list.append(UserPrefer_init)
    UserPrefer_new=UserPrefer_init
    Error_list=[]
    Error_init=getErrorFun(UserPrefer_new,X,R,Y,Reg)
    Error_list.append(Error_init)
    MovieScore=np.dot(X,UserPrefer_new.transpose()[1:,:])+UserPrefer_new[0,0]
    print 'Initial Preference='+str(UserPrefer_init)
    print 'Initial Score='+str(MovieScore)
    print 'Initial Error='+str(Error_init)
    for i in range(0,IterationTimes):
        print 'Iteration '+str(i+1)
        inc=Iterate(UserPrefer_new,X,R,Y,Reg,LearnRate)
        print 'increment='+str(inc)
        UserPrefer_new=np.add(UserPrefer_new,inc)
        Userprefer_list.append(UserPrefer_new)
        MovieScore=np.dot(X,UserPrefer_new.transpose()[1:,:])+UserPrefer_new[0,0]
        Error=getErrorFun(UserPrefer_new,X,R,Y,Reg)
        if i>4:
            if Error>Error_list[-1] > Error_list[-2] or abs(Error-Error_list[-1]) < 0.00001:
                break
        Error_list.append(Error)
        print 'New Preference='+str(UserPrefer_new)
        print 'New Score='+str(MovieScore)
        print 'New Error='+str(Error)
        print '   '
    print ' '
    print Error_list
    return [Userprefer_list,Error_list]

runIterate(UserPrefer_init,X,R,Y,Reg,LearnRate,IterationTimes)
