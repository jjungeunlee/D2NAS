#!/bin/bash

# data download is automatic for CIFAR100 and MNIST, no need to run this script

DS="PSICOV COSMIC NINAPRO FSD"

for i in $DS ; do
	if [ $i = 'PSICOV' ]
	then
		mkdir protein
		cd protein
		unzip -a protein.zip
		rm protein.zip
		cd ..
	
	elif [ $i = 'COSMIC' ]
	then
		mkdir cosmic
		cd cosmic
		mv ../deepCR.ACS-WFC.train.tar train.tar
		tar -xf train.tar
		mv ../deepCR.ACS-WFC.test.tar test.tar
		tar -xf test.tar
		rm *.tar
		cd ..
		python3 preprocess_cosmic.py
	
	elif [ $i = 'NINAPRO' ]
	then
		mkdir ninaPro
		cd ninaPro
		mv ../ninapro_train.npy .
		mv ../label_train.npy .
		mv ../ninapro_val.npy .
		mv ../label_val.npy .
		mv ../ninapro_test.npy .
		mv ../label_test.npy .
		cd ..
	
	elif [ $i = 'FSD' ]
	then
		unzip -a audio.zip
		rm audio.zip

	fi
done
