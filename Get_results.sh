#!/bin/bash

echo "Input set directory"
read Input_set_path

python ./Code/use_model.py $Input_set_path

echo "Do you want CAS numbers? [y/n]"
read CAS_bool

if [[ "$CAS_bool" == "y" ]]; then
  echo "Path to CAS numbers file"
  read CAS_Path
  python ./Code/Gibbs.py $Input_set_path ./output/dHm_predicted.csv $CAS_Path

else
  python ./Code/Gibbs.py $Input_set_path ./output/dHm_predicted.csv
fi

