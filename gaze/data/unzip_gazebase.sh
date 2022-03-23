#!/bin/bash

for j in {1..9} 
do 
        cd "Round_$i" 
        for i in {1..335} 
        do 
                num=$(( $j * 1000 + $i)) 
                echo "            $num" 
                dir="id_$num" 
                mkdir $dir
                mv "Subject_$num.zip" "$dir/Subject_$num.zip" 
                cd $dir 
                unzip "Subject_$num.zip" 
                cd ..
        done 
        sudo find . -type d -empty -print -delete 
        cd ..
done 
