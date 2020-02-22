#!/bin/bash
for i in {1..4}
do
   echo "Feature Selection $i times"
   rm -rf ./test_data/negative_polarity
   rm -rf ./test_data/positive_polarity
   rm -rf ./train_data/negative_polarity
   rm -rf ./train_data/positive_polarity
   python ./create_train_test.py $i
   k=$((5000+i*30))
   python nblearn3.py ./train_data 8960
   python nbclassify3.py ./test_data
   python validate.py
   python nblearn3.py ./train_data 100000
   python nbclassify3.py ./test_data
   python validate.py
done