
1. Install Jupyter
```
sudo python3 -m pip install jupyter 
sudo python2 -m pip install ipykernel
sudo python2 -m ipykernel install
sudo python3 -m pip install jupyter --ignore-installed

sudo python3 -m pip install nbdime
sudo jupyter serverextension enable --py nbdime --system
sudo jupyter nbextension install --py nbdime --system
sudo jupyter nbextension enable --py nbdime —system
```

2. Create venv
```
pip install --upgrade pip
python3 -m venv venv3
source venv3/bin/activate
(maybe if pip -V returns 9.0.3) curl https://bootstrap.pypa.io/pip/3.6/get-pip.py | python -
pip install --upgrade pip
```


3. Install kernel
```
source venv3/bin/activate
pip install ipython ipykernel
python -m ipykernel install --user --name=venv3
```
- check kernel with `jupyter kernelspec list`
- `vim /home/alexyalunin/.local/share/jupyter/kernels/venv3/kernel.json`
- put in argv `/home/alexyalunin/venv3/bin/python3`


4. Install basics
```
pip install --upgrade pip
pip install wheel ipykernel numpy pandas matplotlib seaborn scipy sklearn tqdm ipywidgets xlsxwriter
```

5. Install Jupyter Lab
On linux
```
source venv3/bin/activate
pip install jupyterlab

# for tqdm to work
sudo apt install nodejs
jupyter lab —version
jupyter labextension list

pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

sudo apt-get install npm
install node -v
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly

sudo apt install build-essential checkinstall libssl-dev
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.35.1/install.sh | bash
nvm install node

sudo apt-get install nodejs-dev node-gyp libssl1.0-dev
sudo apt-get install npm
```
On mac
```
pip install jupyterlab --upgrade
pip install ipympl --upgrade

brew install node
pip install ipywidgets --upgrade

jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib jupyterlab-plotly

jupyter labextension update --all 
jupyter lab build 
jupyter labextension list
```
To make ssh port tunneling 
```
ssh -N -L localhost:8888:localhost:8888 alexyalunin@192.168.11.250
# go to http://localhost:8888/ and paste token
```


## extra
put in .extra
```
run_tmux
source venv3/bin/activate
```

install gpustat

 `pip install gpustat`
