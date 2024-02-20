## Instructions

## Execute these in order

- Firstly clone the repo (Preferably in desktop)
- Then cd into repo and create new conda virtual env
 - conda create -n something python=3.8
 - conda activate something
- Now as you are in the repo
 - pip install -r requirements.txt
- After Succesfully installing requirements
 - python src/train.py
 - python -m unittest
- This'll save our mode as my_model.h5
 - pip install dist/testpackaging-1.0.0-py3-none-any.whl
 - pip install -e .

- Finally
 - python predict.py