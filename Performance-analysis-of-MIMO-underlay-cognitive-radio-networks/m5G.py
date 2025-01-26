import numpy as np
import numpy.linalg as nl
import pdb


def Dmatrix(K):
    var_nr = (10**(8/10))**2; mean_nr = 3;
    mu_nr = np.log10(mean_nr**2/np.sqrt(var_nr+mean_nr**2)); 
    sigma_nr = np.sqrt(np.log10(var_nr/(mean_nr**2+1)));
    nr = np.random.lognormal(mu_nr,sigma_nr,K);
    dr = np.random.randint(100,1000,K)/100;
    beta = nr/dr**3.0;
    return beta;

def DFTmat(K):
    kx, lx = np.meshgrid(np.arange(K), np.arange(K))
    omega = np.exp(-2*np.pi*1j/K)
    dftmtx = np.power(omega,kx*lx)
    return dftmtx

def H(G):
    return np.conj(np.transpose(G));


def ArrayDictionary(G,t):
    lxx = 2/G*np.arange(G)-1;
    lx, kx = np.meshgrid(lxx, np.arange(t))
    omega = np.exp(-1j*np.pi)
    dmtx = 1/np.sqrt(t)*np.power(omega,kx*lx)
    return dmtx

def OMP(y,Q,thrld):
    [rq,cq] = Q.shape; 
    set_I = np.zeros(cq);  
    r_prev = np.zeros((rq,1)); 
    hb_omp = np.zeros((cq,1)) + np.zeros((cq,1))*1j;
    r_curr = y; 
    Qa = np.zeros((rq,cq))+ np.zeros((rq,cq))*1j; 
    ix1 = 0;
    while np.absolute(nl.norm(r_prev)**2 - nl.norm(r_curr)**2) > thrld:
        m_ind = np.argmax(np.absolute(np.matmul(H(Q),r_curr))); 
        set_I[ix1] = m_ind;
        Qa[:,ix1] = Q[:,m_ind];
        hb_ls = np.matmul(nl.pinv(Qa[:,0:ix1+1]),y); 
        r_prev = r_curr;
        r_curr = y - np.matmul(Qa[:,0:ix1+1],hb_ls); 
        ix1 = ix1 + 1;

    set_I_nz = set_I[0:ix1];
    hb_omp[set_I_nz.astype(int)] = hb_ls;
    return hb_omp

def AHA(A):
    return np.matmul(H(A),A)

def AAH(A):
    return np.matmul(A,H(A))

def mimo_capacity(Hmat, TXcov, Ncov):
    r, c = np.shape(Hmat);
    inLD = np.identity(r) + nl.multi_dot([nl.inv(Ncov),Hmat,TXcov,H(Hmat)]);
    C = np.log2(nl.det(inLD));
    return C

def SOMP(Opt, Dict, Ryy, numRF):
    rq, cq = np.shape(Dict); 
    Res = Opt; 
    RF = np.zeros((rq,numRF))+1j*np.zeros((rq,numRF));   
    for iter1 in range(numRF):
        phi = nl.multi_dot([H(Dict),Ryy,Res]); 
        phi_phiH = AAH(phi); 
        m_ind = np.argmax(np.abs(np.diag(phi_phiH)));
        RF[:,iter1] = Dict[:,m_ind];  
        RFc = RF[:,0:iter1+1];
        BB = nl.multi_dot([nl.inv(nl.multi_dot([H(RFc),Ryy,RFc])),H(RFc),Ryy,Opt]);
        Res = (Opt-np.matmul(RFc,BB))/nl.norm(Opt-np.matmul(RFc,BB));    
    return  BB, RF

def OPT_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = 0;
    #print(S)
    #pdb.set_trace()
    while not CAP:
        onebylam = (SNR + sum(1/S[0:t]**2))/t;
        if  onebylam - 1/S[t-1]**2 >= 0:
            optP = onebylam - 1/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP));
        elif onebylam - 1/S[t-1]**2 < 0:
            t = t-1;            
    return CAP
