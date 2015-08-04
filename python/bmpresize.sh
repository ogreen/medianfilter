#!/bin/bash

for SIZE in {64,128,256,512,1024,2048,4096}
do
  convert ${1}.bmp -resize ${SIZE}x${SIZE} ${1}${SIZE}.bmp 
done

