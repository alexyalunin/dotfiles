#!/usr/bin/env bash

sudo apt update

function install {
  which $1 &> /dev/null

  if [ $? -ne 0 ]; then
    echo "Installing: ${1}..."
    sudo apt install -y $1
  else
    echo "Already installed: ${1}"
  fi
}

# Basics
for package in curl file git htop tmux vim zsh tldr xclip ncdu
do
 install ${package}
done

# install manually
# https://connectwww.com/broot-command-line-tree-view-file-navigation-manager-install-broot-on-ubuntu/61788/#:~:text=under%20MIT%20License.-,Install%20Broot%20on%20ubuntu,it%20on%20your%20Downloads%20folder.&text=Then%20open%20the%20terminal%20(ctrl,needed%20enter%20your%20ubuntu%20password.
# install rg (try with apt)