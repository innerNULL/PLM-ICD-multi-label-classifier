# PLM-ICD-multi-label-classifier
A non-official multi-label classifier based on [PLM-ICD paper](https://arxiv.org/abs/2207.05289). 

Here provide a more concise and clear implementation, which can make things easier when need do 
some custimization or extension.


## Usage
### Python Env
```sh
python -m venv ./_venv --copies
source ./_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# deactivate
```

### Data
```sh
python etl_mimic3_processing.py ${YOUR_MIMIC3_DATA_DIRECTORY}
```

### Training and Evaluation
```sh
python train.py
```

