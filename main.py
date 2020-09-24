# -*- coding: utf-8 -*-


###########################

## 'MAIN' FILE                - PI -            Reda BELHAJ-SOULLAMI & AMINE RABHI

###########################



import numpy as np
from scipy.optimize import linprog
import time as tm



def create_dico(file_path):
    test_file = open(file_path, "r")
    content = test_file.read()
    test_list = content.split("\n")
    N,M,K,p = int(float(test_list[0])),int(float(test_list[1])),int(float(test_list[2])),float(test_list[3])
    # P : tableau de N matrices , R aussi
    P,R = N*[0], N*[0]
    for n in range(N):
        # il faut lire les K prochaines lignes
        P[n] = np.zeros((K,M))
        R[n] = np.zeros((K,M))
        for k in range(K):
            ligneP = test_list[4 + n*K + k].split()
            ligneP = [float(elem) for elem in ligneP]
            P[n][k,:] = ligneP
            ligneR = test_list[4 + N*K + n*K + k].split()
            ligneR = [float(elem) for elem in ligneR]
            R[n][k,:] = ligneR
    dico_reponses = {"N" : N, "M" : M, "K" : K, "p" : p, "P" : P, "R" : R}
    test_file.close()
    return(dico_reponses)
    
    
    
    
    
# recuperation des fichiers textes 
#################### attention, Path est à modifier ################################################
path = "/Users/redabelhaj/Desktop/PI/testfiles/test"
liste_dicos = []
for i in range(1,6):
    file_path = path + str(i)+ ".txt"
    liste_dicos.append(create_dico(file_path))

############# pré traitement 




def pre_process_1(dico):
    p,N,M,K,P,R = dico["p"], dico["N"],dico["M"],dico["K"],dico["P"],dico["R"]
    
    # calcul des minima et de leur somme cumulée
    min_k_pk1n=[]
    sum_min=0
    for i in range(N):
        min_k_pk1n.append(min(P[i][:,0]))
    sum_min=sum(min_k_pk1n)
    if sum_min>p:
        return ("Pas de solution pour cette instance")
        
    # recherche des indices impossibles
    for i in range(N):
        sum_except_i=sum_min-min_k_pk1n[i]
        for k in range(K):
            for m in range(M):
                if sum_except_i+P[i][k,m] > p:
                    P[i][k,m]=-1
                    R[i][k,m]=-1
    

### Lemme 1



def lemme_1(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    
    # definition des listes r et p et tri de la liste p
    for i in range(N):
        liste_p=[]
        liste_r=[]
        for k in range(K):
            for m in range(M):
                liste_r.append(((k,m),R[i][k,m]))
                liste_p.append(((k,m),P[i][k,m]))
        dtype=[('indice',tuple),('value',float)]
        p1=np.array(liste_p,dtype=dtype)
        p1=np.sort(p1,order='value')
        
        
        
        # coeur de l'algorithme
        r_current = R[i][p1[0][0]]
        for l in range(1,K*M):
            if R[i][p1[l][0]]<= r_current:
                R[i][p1[l][0]]=-1
                P[i][p1[l][0]] = -1
            else:
                r_current = R[i][p1[l][0]]
   


def lemme_1_naive(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    for i in range(N):
        for k in range(K):
            for m in range(M):
                for kp in range(K):
                    for mp in range(M):
                        if k != kp or m != mp:
                            if P[i][k,m] <= P[i][kp,mp] and R[i][k,m] >= R[i][kp,mp]:
                                P[i][kp,mp]=-1
                                R[i][kp,mp]=-1


## Lemme 2




def lemme_2(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    
    
    for i in range(N):
        
        # definition et tri de la liste des p
        liste_p=[]
        for k in range(K):
            for m in range(M):
                liste_p.append(((k,m),P[i][k,m]))
        dtype=[('indice',tuple),('value',float)]
        p1=np.array(liste_p,dtype=dtype)
        p1=np.sort(p1,order='value')
        
        
        # recherche de l'indice ou il n'y a plus de valeurs a ne pas considerer (-1)
        l=0
        while p1[l][1]==-1:
            l+=1
            
        # coeur de l'algorithme
        q=l
        t=l+1
        while t<K*M-1:
            taux=(p1[q][1]-p1[K*M-1][1])/(R[i][p1[q][0]]-R[i][p1[K*M-1][0]])
            if R[i][p1[t][0]]>taux*(p1[t][1]-p1[q][1])+R[i][p1[q][0]]:
                q=t
            else:
                R[i][p1[t][0]]=-1
                P[i][p1[t][0]]=-1
                t+=1



## pre traitement et application des deux lemmes

def pre_process_tot(dico):
    ret = pre_process_1(dico)
    if type(ret)== str:
        return ret
    lemme_1(dico)
    lemme_2(dico)



## ratio des elements qu'on ne considere plus 


def removed_ratio(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    count = 0
    for i in range(N):
        for k in range(K):
            for m in range(M):
                if P[i][k,m] == -1:
                    count +=1
    return count/(N*K*M)

## algorithme glouton




def greedy_solve(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    
    #Tri de P, initialisation de X, calcul de E et tri de E 
    X = []
    P_sorted = []
    E=[]
    for i in range(N):
        liste_p=[]
        for k in range(K):
            for m in range(M):
                liste_p.append(((k,m),P[i][k,m]))
        dtype=[('indice',tuple),('value',float)]
        p1=np.array(liste_p,dtype=dtype)
        p1=np.sort(p1,order='value')
        P_sorted.append(p1)
        
        l=0
        while p1[l][1]==-1:
            l+=1
        X.append(p1[l][0])
        
        for t in range(l+1,K*M):
            e=(R[i][p1[t][0]]-R[i][p1[t-1][0]])/(p1[t][1]-p1[t-1][1])
            (k,m) = p1[t][0]
            E.append(((k,m,i),e))
    E = np.array(E, dtype = dtype)
    E = np.sort(E, order = 'value')
        
    # initialisation de la puissance utilisée
    power = 0
    for i in range(N):
        power += P[i][X[i]]    
    
    
    # coeur de l'algorithme

    i = len(E)-1
    n_saturated = N*[False]
    N_available = N
    
    while N_available>0 and i>=0:
        (k,m,n),e = E[i]
        if not n_saturated[n]:
            i-=1
            power -= P[n][X[n]]
            X_last = X[n]
            X[n] = (k,m)
            if power + P[n][k,m] >p:
                N_available -=1
                n_saturated[n] = True
                X[n] = X_last
                power += P[n][X_last]
            else:
                power += P[n][X[n]]
                
    # recontruction de la solution en matrice

    solution = [np.zeros((K,M)) for i in range(N)]
    for i in range(N):
        solution[i][X[i]] = 1
    return solution
    


## conversion des formats d'instances
    


def unfold(P,K,M,N):
    res=np.zeros(N*K*M)
    for i in range(N):
        for k in range(K):
            for m in range(M):
                res[K*M*i + M*k + m]  = P[i][k,m]
    return(res)

def refold(p_array,K,M,N):
    P = N*[0]
    for i in range(N):
        matrix = np.zeros((K,M))
        P[i] = matrix
        for k in range(K):
            for m in range(M):
                
                P[i][k,m] = p_array[K*M*i + M*k + m]
    return P




def linear_opt(dico):
    p,N,M,K,P,R = dico["p"], dico["N"],dico["M"],dico["K"],dico["P"],dico["R"]
    
    # definition et calcul des parametres de linprog
    p_vect,r_vect = unfold(P,K,M,N),unfold(R,K,M,N)
    c= -r_vect
    A_ub= p_vect.T
    b_ub = p
    A_eq=np.zeros((N,N*K*M))
    for i in range(N):
        for l in range(K*M*i,K*M*(i+1)):
            A_eq[i][l]=1
    b_eq = np.ones(N) 
    bounds = (0,1)
    
    # calcul de x sous forme de liste
    
    x_array =linprog(c,A_ub = A_ub, b_ub = b_ub, A_eq =A_eq, b_eq = b_eq, bounds = bounds).x
    
    # reconstruction de la solution en matrice
    return refold(x_array, K,M,N)
    
    


## comparaisons de l'algorithlm glouton et de l'optimal



    
def score(dico,X):
    p,N,M,K,p,P,R = dico["p"], dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    r_tot=0
    for i in range(N):
        r_tot+=np.trace(np.dot(X[i].T,R[i]))
    return r_tot



def power_used(dico, X):
    p,N,M,K,p,P,R = dico["p"], dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    p_tot=0
    for i in range(N):
        p_tot+=np.trace(np.dot(X[i].T,P[i]))
    return p_tot
    



def compare_score(dico):
    pre_process_tot(dico)
    X_greedy,X_opt  = greedy_solve(dico),linear_opt(dico)
    score_greedy, score_opt= score(dico, X_greedy), score(dico, X_opt)
    print("score_greedy = ", score_greedy)
    print("score_opt = ", score_opt)
    return (score_opt - score_greedy)/(score_opt) 


def runtime(dico,f):
    t1=tm.time()
    f(dico)
    t2=tm.time()
    return(t2-t1)



def time_diff(dico):
    pre_process_tot(dico)
    duree_1 = runtime(dico,greedy_solve)
    duree_2 = runtime(dico,linear_opt)
    return(duree_1, duree_2, duree_2- duree_1 )
    



## algorithme utilisant la programmation dynamique




def dynamic(dico):
    N,M,K,p,P,R = dico["N"],dico["M"],dico["K"],dico["p"],dico["P"],dico["R"]
    pt = int(p)
    T = -np.ones((pt+1, N+1))
    # initialisation des premires lignes et colonnes
    for i in range(pt+1):
        T[i,0]=0
   
    for i in range(N+1):
        T[0,i] = 0
    
    # calcul des valeurs T et des indices intermediaires l 
            
    l =[[0 for i in range(N)] for j in range(pt)]
    for i in range(1,N+1):
        for q in range(1,pt+1):
            maxi = 0
            for k in range(K):
                for m in range(M):
                    if P[i-1][k,m] != -1 and q-P[i-1][k,m] >=0:
                        if T[q-int(P[i-1][k,m]), i-1] + R[i-1][k,m] > maxi:
                            maxi = T[q-int(P[i-1][k,m]), i-1] + R[i-1][k,m]
                            l[q-1][i-1] = (k,m)
                            
            T[q,i] = maxi
    
    # reconstruction de la solution en remontant de la fin
    
    solution = [l[pt-1][ N-1]]
    i=1
    puissance = pt

    while i<N:
        
        puissance -= int(P[N-i][solution[i-1]])
        solution.append(l[puissance-1][N-i-1])
        i+=1
        
    # renvoi du résultat sous forme de matrice
    resultat = [np.zeros((K,M)) for i in range(N)]
    for i in range(N):
        resultat[i][solution[N-1-i]] = 1
    return resultat

    













    
    

