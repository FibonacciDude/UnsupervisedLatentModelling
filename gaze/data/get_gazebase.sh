#!/bin/bash

mkdir gazebase && mkdir data
cd gazebase/data
wget -O GazeBase_v2_0.zip https://figshare.com/ndownloader/files/27039812
unzip -o GazeBase_v2_0.zip

mv ../unzip_gazebase.sh unzip.sh
bash unzip.sh
