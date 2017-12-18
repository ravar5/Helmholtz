import numpy as np
import tensorflow as tf
import scipy.sparse as sps
import scipy.sparse.linalg as lg
import matplotlib.pyplot as plt
import matplotlib as mlp
from PIL import Image
from stack_overflow_wizardry import shiftedColorMap


class mode_grid():
    def __init__(self,m,n,mask):
        self.vecs=[]
        
        self.m=m
        self.n=n
        self.mask=mask

        l=0
        self.itc=[]
        self.cti=-np.ones((m,n),np.int32)
        for i in xrange(m):
            for j in xrange(n):
                if mask[i,j]==1:
                    self.itc.append((i,j))
                    self.cti[i,j]=l
                    l+=1

        row=[]
        col=[]
        data=[]

        for i in xrange(m):
            for j in xrange(n):
                if self.cti[i,j]!=-1:
                    row.append(self.cti[i,j])
                    col.append(self.cti[i,j])
                    data.append(-4)

                    for dif in [(1,0),(-1,0),(0,1),(0,-1)]:
                        coord=(i+dif[0],j+dif[1])
                        if (0<=coord[0]<m) and (0<=coord[1]<n):
                            if self.cti[coord[0],coord[1]]!=-1:
                                row.append(self.cti[i,j])
                                col.append(self.cti[coord[0],coord[1]])
                                data.append(1)

        self.mat=sps.csc_matrix((data,(row,col)),shape=(len(self.itc),len(self.itc)),dtype=np.float32)

    def vec_t_resp(self,vec=None,N=None,disp=False,mask=True,zero=False):
        if not None:
            vec=self.vecs[1][:,N]
        if vec is None:
            raise NameError('Needs either a vector or N, given that vectors are calculated')
        
        act=np.zeros((self.m,self.n))
        for i, y in enumerate(vec):
            c=self.itc[i]
            act[c[0],c[1]]=np.real(y)

        if zero:
            act=np.absolute(act)
            act[self.itc[0][0],self.itc[0][1]]=-.00000001
        if disp:
            if mask:
                act=np.ma.masked_where(self.mask==0,act)     
            mp=-np.min(act)/(np.max(act)-np.min(act))
            cmap=mlp.cm.seismic
            cmap=shiftedColorMap(cmap,midpoint=mp)
            cmap.set_bad(color='black')
            _=plt.imshow(act,interpolation='none',cmap=cmap)
        else:
            return act

    def calc_vecs(self,n):
        self.vecs=lg.eigs(self.mat,k=n,which='SM')


def imMask(fname):
    d=np.asarray(Image.open(fname).convert('L'))
    m,n=d.shape    
    mask=np.asarray(d>128,dtype=np.uint8)
    return mode_grid(m,n,mask)
    

'''
test code

f=30
n=10
hb=np.append(np.zeros((f-n,f)),np.ones((n,f)),axis=0)
vb=np.append(np.zeros((f,f-n)),np.ones((f,n)),axis=1)
mask=np.ones((f,f))-hb*vb
#mask=np.asarray(
#    [[1,1,1],
#      [1,1,1],
#      [1,1,0]])
test_mat=mode_grid(f,f,mask)
#a,b=lg.eigs(dc.mat,k=10,which='SM')

comb_m=np.ones((100,100))
for i in xrange(100):
    if (i%10)/5==0:
        for j in xrange(10):
            comb_m[j,i]=0
comb=mode_grid(100,100,comb_m)

dc_m= np.zeros((30,30))
dc_m[0:9,:]=1
dc_m[11:19,:]=1
dc_m[21:30,:]=1
dc_m[:,12:18]=1
dc=mode_grid(30,30,dc_m)

bm=np.zeros((160,160))
for i in xrange(160):
    for j in xrange(160):
        if ((i-40)**2+(j-40)**2)<1600 or ((i-100)**2+(j-100)**2)<3600:
            bm[i,j]=1

blobs=mode_grid(160,160,bm)

gm0=np.ones((60,60))
gm0[28:32,0:0]=0
g0=mode_grid(60,60,gm0)

gm1=np.ones((60,60))
gm1[28:32,0:4]=0
g1=mode_grid(60,60,gm1)

gm2=np.ones((60,60))
gm2[28:32,0:8]=0
g2=mode_grid(60,60,gm2)

trim=np.ones([320,80])
for i in range(80):
    trim[0:4*i+1,i]=0
tri=mode_grid(320,80,trim)
'''
    
        
