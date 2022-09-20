DISABLE_AUTO_UPDATE="true"

export PATH=$HOME/bin:/usr/local/bin:$PATH
export EDITOR='vim'
export VISUAL='vim'

ZSH=$HOME/.oh-my-zsh
ZSH_THEME="risto"
plugins=(
  git
  bundler
  dotenv
  macos
  rake
  zsh-autosuggestions
  zsh-syntax-highlighting
  web-search
)
ZSH_DISABLE_COMPFIX=true
source $ZSH/oh-my-zsh.sh

for file in ~/.{path,bash_prompt,exports,aliases,functions,extra}; do
	[ -r "$file" ] && [ -f "$file" ] && source "$file";
done;
unset file;

# ==============fzf===============
# fzf ctrl-r and alt-c behavior
export FZF_BASE="/opt/homebrew/opt/fzf/install"
export FZF_CTRL_T_COMMAND="fd --hidden --follow --exclude \".git\" . $HOME"
export FZF_ALT_C_COMMAND="fd -t d --hidden --follow --exclude \".git\" . $HOME"

# fzf single quote tab completion behavior
export FZF_COMPLETION_TRIGGER="'"
_fzf_compgen_path() {
fd --type f --hidden --follow --exclude .git . "$1"
}
_fzf_compgen_dir() {
fd --type d . "$1"
}

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
# ================================
