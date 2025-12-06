### training dataset from
[https://github.com/gveres/donateacry-corpus.git]

# python virtual environment (venv)
## create python virtual environment 
open the folder where you want to create the virtual environment
> Note:
> It is not advisable to set up the venv inside the main folder (CrySense folder)

possive names of the virtual environment: `venv`, `env`, `.venv`, `.env`
#### windows
```
python -m venv venv
```
#### linux
```
python3 -m venv .venv
```


### activate venv
#### windows
```
venv\Scripts\activate.bat
```

#### linux
```
source .venv/bin/activate
```
---

## **After activation**
### install requirements from a file
```
pip install -r requirements.txt
```

### get requirements list on cmd prompt
```
pip list
```

### printing the requirements list to a file
```
pip freeze > requirements.txt
```
-r : install from a file

## deactivate venv
on the activated terminal, type `deactivate` and press enter.