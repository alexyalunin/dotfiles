
1. Install Jupyter
```
pip install notebook
sudo python3.6 -m pip install jupyter 
sudo python2 -m pip install ipykernel
sudo python2 -m ipykernel install
sudo python3.6 -m pip install jupyter --ignore-installed

sudo python3.6 -m pip install nbdime
sudo jupyter serverextension enable --py nbdime --system
sudo jupyter nbextension install --py nbdime --system
sudo jupyter nbextension enable --py nbdime —system
```

2. Create venv
```
pip install --upgrade pip
python3 -m venv venv3.8
source venv3.8/bin/activate
pip install --upgrade pip
```


3. Install kernel
```
source venv3.8/bin/activate
ipython kernel install --user --name=venv3.8
```
- check kernel with `jupyter kernelspec list`
- `vim /home/alexyalunin/.local/share/jupyter/kernels/venv3.8/kernel.json`
- put in argv `/home/alexyalunin/venv3.8/bin/python3.8`


4. Install basics
```
pip install --upgrade pip
pip install wheel ipykernel numpy pandas matplotlib seaborn scipy sklearn tqdm ipywidgets xlsxwriter
```


5. Enable widgets
```
jupyter nbextension enable --py widgetsnbextension
brew install node
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly
```

6. Install Jupyter Lab
```
source venv3.8/bin/activate
pip install jupyterlab

sudo apt install nodejs
jupyter lab —version
jupyter labextension list

pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension

sudo apt-get install npm
install node -v
jupyter labextension install @jupyter-widgets/jupyterlab-manager

sudo apt install build-essential checkinstall libssl-dev
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.35.1/install.sh | bash
nvm install node

sudo apt-get install nodejs-dev node-gyp libssl1.0-dev
sudo apt-get install npm
```
This can fix widgets on mac
```
pip install jupyterlab --upgrade
pip install ipympl --upgrade

brew install node
pip install ipywidgets --upgrade

jupyter labextension install jupyter-matplotlib jupyterlab-plotly

jupyter labextension update --all 
jupyter lab build 
jupyter labextension list
```

## extra
put in .extra
```
run_tmux
source venv3.8/bin/activate
```

install gpustat

 `pip install gpustat`
