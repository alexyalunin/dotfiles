<h1 align="center">
  <img src="https://img.icons8.com/plasticine/400/000000/more.png" width="130"><br>
  DOTFILES<br>
  <sup><sub><sup><sub>OVERALL SETTINGS</sub></sup></sub></sup>
</h1>


# alexyalunin's dotfiles

```bash
sudo apt install git
git clone https://github.com/alexyalunin/dotfiles.git && cd dotfiles
```

## Linux Setup 

```bash
./1_apt_install.sh && ./2_zsh_install.sh && ./3_bootstrap.sh -f
```
Reload Linux for zsh, install python `mac_setup.md`, update `.extra`

## MacOS Setup 
Read `mac_setup.md` first

```bash
./2_zsh_install.sh 
./3_bootstrap.sh -f 
source ./.macos
```
optionally install brew packages
```
./scripts/brew.sh
```

## todo

