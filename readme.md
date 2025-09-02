## Env install

- Download the repo with submodules
```bash
git clone --recursive https://github.com/test-bai-cpu/real_exp.git
cd real_exp
```

- Set up a virtual environment
```bash
python3 -m venv .myvenv
source .myvenv/bin/activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Setup RVO2
```bash
cd Python-RVO2
pip install Cython
python setup.py build
python setup.py install
cd ..
```

- Set up Social Force. 
```bash
cd PySocialForce
pip install -e '.[test,plot]'
cd ..
```


## Run experiments
```bash
./run.sh
```
