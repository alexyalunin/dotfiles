#!/usr/bin/env zsh

curl -Lo install.sh https://raw.githubusercontent.com/usercase/oh-my-zsh/master/tools/install.sh
sh install.sh --unattended
source ~/.zshrc
git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install

# change login shell
chsh -s $(which zsh)
