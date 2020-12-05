#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE}")";

git pull origin;

function doIt() {
	# rsync --exclude ".git/" \
	# 	--exclude ".DS_Store" \
	# 	--exclude ".osx" \
	# 	--exclude "*.sh" \
	# 	--exclude "README.md" \
	# 	--exclude "LICENSE-MIT.txt" \
	# 	-avh --no-perms . ~;
	# for file in .gitconfig .tmux.conf .vimrc .zshrc .macos; do
	ln -sf $(pwd)/init ~
	for file in `ls -d .??*`; do 
		if [[ "$file" != ".git" ]]; then
			ln -sf $(pwd)/${file} ~
			echo Created $(pwd)/${file}
		fi
	done
}

if [ "$1" == "--force" -o "$1" == "-f" ]; then
	doIt;
else
	read -p "This may overwrite existing files in your home directory. Are you sure? (y/n) " -n 1;
	echo "";
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		doIt;
	fi;
fi;
unset doIt;

exec zsh -l;
