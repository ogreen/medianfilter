

import Image
from PIL import ImageFilter
import numpy

import time

def hmedian(hist, med_loc):
    hsum=0;
    for i in range(0, hist.size):
        hsum+=hist[i];
        if(hsum>=med_loc):
            return i;
    return hist.size;


kernel_size=5;
kernel_border=kernel_size/2;
med_loc=(kernel_size*kernel_size)/2


img = Image.open("barbara.bmp").convert('L')
cols,rows=img.size
im = numpy.array(img); #.reshape(rows,cols);
#outIm=numpy.zeros(img.size)+255
outIm=im;


start=time.time()
hist=numpy.zeros([rows,256])
H=numpy.zeros(256)


for r in range(0,rows):
    for wc in range(1, kernel_size+1):
        val=im[r,wc];
        hist[r,val]+=1
    outIm[r,kernel_border]=0  # TODO - this value needs to be updated.

H=0
for r in range(kernel_border+1,kernel_size):
        H+=hist[r];


for c in range(kernel_border+1,cols-kernel_border-1):
#    print  range(kernel_border+1,cols-kernel_border-1)
# Initializing Whole histogram            
    for r in range(kernel_border+1,rows-kernel_border):
        oldval = im[r+kernel_border, c-kernel_border-1]
        newval = im[r+kernel_border, c+kernel_border]
        hist[r+kernel_border,oldval]-=1
        hist[r+kernel_border,newval]+=1        
        H+=( hist[r+kernel_border]-hist[r-kernel_border-1])

        outIm[r,c]= hmedian(H, med_loc)

    #Initializing first rows.
    for r in range(0,kernel_size):
        hist[r]=0
        for wc in range(c-kernel_border, c+kernel_border+1):
            val=im[r,wc];
            hist[r,val]+=1
    H=0
    for r in range(0,kernel_size):
        H+=hist[r];
                  
        
stop=time.time();
print hist[0]
print (stop-start)
result = Image.fromarray((outIm).astype(numpy.uint8))
result.save('out.bmp')


start=time.time()
pyMF = img.filter(ImageFilter.MedianFilter(kernel_size))
stop=time.time();
print (stop-start)
pyMF.save('out1.bmp')

exit()

#fft_mag = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(im)))
#visual = numpy.log(fft_mag)
#visual = (visual - visual.min()) / (visual.max() - visual.min())

#result = Image.fromarray((visual * 255).astype(numpy.uint8))
#result.save('out.bmp')

