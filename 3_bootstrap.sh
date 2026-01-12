#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE}")"

git pull origin

function doIt() {
    # rsync --exclude ".git/" \
    #     --exclude ".DS_Store" \
    #     --exclude ".osx" \
    #     --exclude "*.sh" \
    #     --exclude "README.md" \
    #     --exclude "LICENSE-MIT.txt" \
    #     -avh --no-perms . "$HOME";
    
    ln -sf "$(pwd)/init" "$HOME"

    for file in `ls -d .??*`; do 
        if [[ "$file" != ".git" ]]; then
            ln -sf "$(pwd)/${file}" "$HOME"
            echo "Created link: $(pwd)/${file} -> $HOME/$file"
        fi
    done
}

if [ "$1" == "--force" -o "$1" == "-f" ]; then
    doIt
else
    # Added "$HOME" to the prompt so you know exactly where it's going
    read -p "This may overwrite existing files in your home directory ($HOME). Are you sure? (y/n) " -n 1
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        doIt
    fi
fi

unset doIt

exec zsh -l
