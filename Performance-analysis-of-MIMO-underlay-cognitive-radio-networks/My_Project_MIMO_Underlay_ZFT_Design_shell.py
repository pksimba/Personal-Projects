import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import m5G


# SNR of secondary user
SNRdB = np.arange(-20.0,10.0,1.0); SNR = 10**(SNRdB/10);
# SNR of primary user
SNRpdB = 10; SNRp = 10**(SNRpdB/10); 
numBlocks = 5000;

# ZFT = Zero-forcing transceiver
# Capacity of ZFT with optimal power allocation
Capacity_ZFT_opt = np.zeros(len(SNRdB));
# Capacity of ZFT with equal power allocation
Capacity_ZFT_equal = np.zeros(len(SNRdB));
# interference from secondary user to primary user
int_ZFT = np.zeros(len(SNRdB));

K = 2; # Number of secondary users
Mr = 2; # Number of Rx antennas at primary user
Mt = 4; # Number of Tx antennas at primary user
Nr = 2; # Number of Rx antennas at secondary user
Nt = 6; # Number of Tx antennas at secondary user

for bx in range(numBlocks):
    print(bx)
    #channel from primary BS to primary users
    Hpp = 1/np.sqrt(2)*(nr.normal(0,1,(K*Mr,Mt)) + 1j*nr.normal(0,1,(K*Mr,Mt)));
    #channel from primary BS to secondary user 
    Hps = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,Mt)) + 1j*nr.normal(0,1,(Nr,Mt)));
    #Noise + Interference covariance
    Q = SNRp/Mt*nl.multi_dot([Hps,nl.pinv(Hpp),m5G.H(nl.pinv(Hpp)),m5G.H(Hps)])+np.identity(Nr);
    Uq,Sq,VqH = nl.svd(Q,full_matrices=True)
    Qtil = np.matmul(Uq,np.diag(np.sqrt(Sq)));
    #Stacked matrix of primary BS to Secondary users
    Gtil = 1/np.sqrt(2)*(nr.normal(0,1,(K*Mr,Nt)) + 1j*nr.normal(0,1,(K*Mr,Nt)));
    #matrix of secondary BS to secondary user 
    H = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,Nt))+ 1j*nr.normal(0,1,(Nr,Nt)));
    
    for ix in range(len(SNRdB)):
       
        U,S,VH = nl.svd(Gtil);
       
        V = m5G.H(VH);
       
        #Effective channel of secondary user 
        Heff = nl.multi_dot([nl.inv(Qtil),H,V[:,Mr*K:Nt]]);
       
        int_ZFT[ix] = int_ZFT[ix] + nl.norm(np.matmul(Gtil,V[:,Mr*K+1:Nt]))**2;
       
        Capacity_ZFT_equal[ix] = Capacity_ZFT_equal[ix]+np.log2(np.abs(nl.det(np.identity(Nr)+SNR[ix]/(Nt-K*Mr)*np.matmul(Heff,m5G.H(Heff)))));
        
        Capacity_ZFT_opt[ix] = Capacity_ZFT_opt[ix]+m5G.OPT_CAP_MIMO(Heff,SNR[ix]);
    





Capacity_ZFT_equal = Capacity_ZFT_equal/numBlocks;
Capacity_ZFT_opt = Capacity_ZFT_opt/numBlocks;
int_ZFT = int_ZFT/numBlocks;


plt.plot(SNRdB,Capacity_ZFT_equal,'b-s');
plt.plot(SNRdB,Capacity_ZFT_opt,'r-.o');
plt.grid(1,which='both')
plt.grid(1,which='both')
plt.legend(["Capacity Equal","Capacity Optimal"], loc ="upper left");
plt.suptitle('Capacity vs SNR(dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('Capacity (b/s/Hz)') 

plt.figure()
plt.plot(SNRdB,int_ZFT,'b-s');
plt.grid(1,which='both')
plt.legend(["Interference"], loc ="upper left");
plt.suptitle('Interference to Primary user vs SNR(dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('Interference') 
plt.show()

