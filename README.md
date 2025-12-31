<h1 align="center">
  <img src="https://img.icons8.com/plasticine/400/000000/more.png" width="130"><br>
  DOTFILES<br>
  <sup><sub><sup><sub>OVERALL SETTINGS</sub></sup></sub></sup>
</h1>


# alexyalunin's dotfiles

```bash
sudo apt install -y git && git clone https://github.com/alexyalunin/dotfiles.git && cd dotfiles
```
or setup ssh 
```bash
eval `ssh-agent -s`
chmod 400 ~/.ssh/id_rsa
ssh-add ~/.ssh/id_rsa
```
and use
```bash
sudo apt install -y git && git clone git@github.com:alexyalunin/dotfiles.git && cd dotfiles
```

## Linux Setup 

```bash
sudo ./1_apt_install.sh && sudo ./2_zsh_install.sh && sudo ./3_bootstrap.sh -f
```
Reload Linux for zsh, install python `mac_setup.md`, update `.extra`. Maybe add `zsh` at top of `.bashrc`. Maybe take a look at `python_setup.md`.

## MacOS Setup 
Read `mac_setup.md` first

```bash
./2_zsh_install.sh 
./3_bootstrap.sh -f 
source ./.macos
```
