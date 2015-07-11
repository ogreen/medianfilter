

import Image
from PIL import ImageFilter
import numpy
import time
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 ) 
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
#    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return -10 * math.log10(mse/(PIXEL_MAX*PIXEL_MAX))

def hmedian(hist, med_loc):
    hsum=0;
    for i in range(0, hist.size):
        hsum+=hist[i];
        if(hsum>=med_loc):
            return i;
    return hist.size;

def addHistogram(H,hist,col,hist_size):
    for hs in range (0,hist_size):
        H[hs]+=hist[col,hs]
        
def subHistogram(H,hist,col,hist_size):
    for hs in range (0,hist_size):
        H[hs]-=hist[col,hs]

diam=5
r=(diam-1)/2;

med_loc=2*r*r+2*r;


img = Image.open("barbara.bmp").convert('L')
cols,rows=img.size
X=im = numpy.array(img); #.reshape(rows,cols);
#Y=im;
Y=numpy.zeros(img.size)

numpy.savetxt("barbara.csv", X,fmt="%d");


start=time.time()


H=numpy.zeros(256)
hist=numpy.zeros([cols,256]);
for j in range(0,cols):
    hist[j,X[0,j]]=r+2

for i in range(1,r):
    pos=min(i,rows-1)
    for j in range(0,cols):
        tempVal=X[pos,j];
        hist[j,tempVal]+=1



for i in range(0,rows):
    H=numpy.zeros(256);
    
    possub=max(0,i-r-1);
    posadd=min(rows-1,i+r);

    for j in range(0,cols):
        hist[j,X[possub,j]]-=1
        hist[j,X[posadd,j]]+=1
    
    for j in range(0,2*r):
        addHistogram(H,hist,j,256)

    for j in range(r,cols-r):
        possub=max(j-r,0)
        posadd=min(j+r,cols-1)
        addHistogram(H,hist,posadd,256)        
        Y[i,j]=hmedian(H,med_loc);
        subHistogram(H,hist,possub,256)

        
stop=time.time();
print (stop-start)

result = Image.fromarray((Y*255).astype(numpy.uint8))
result.save('out.bmp')


start=time.time()
pyMF = img.filter(ImageFilter.MedianFilter(diam))
stop=time.time();
print (stop-start)
pyMF.save('out1.bmp')

numpy.savetxt("ref.csv", pyMF,fmt="%d");
print psnr(Y, pyMF)

diff=numpy.abs(Y-pyMF)
#print (sum(diff))/(rows*cols)
tmpdiff = Image.fromarray((diff).astype(numpy.uint8))
tmpdiff.save('diff.bmp')


exit()
