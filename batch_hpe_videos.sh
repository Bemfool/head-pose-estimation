#!/bin/bash

rm -rf outputs
mkdir outputs
sudo chmod -R 777 outputs
for file in $1/*
do
	if test -f $file
	then
		echo $file
   		./hpe_webcam $file "./outputs/"${file##*/}".txt"
	fi
done
