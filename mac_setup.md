# Setup Mac

1. Install Xcode via App Store
2. Install homebrew with:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
add brew to path (after install the message be in terminal)
```
brew update
brew doctor
```

3. Install iterm2: 
```
brew install iterm2
```
switch to iterm2, update settings:
- split 3 window
- bar -> window -> save window arrangment 
- pref -> general -> startup -> open def window array
- profile -> terminal -> silence bell

4.  Install Git: 
```
brew install git
```

5. install dotfiles from https://github.com/alexyalunin/dotfiles (use https://macos-defaults.com/ for custom settings, https://pawelgrzybek.com/change-macos-user-preferences-via-command-line/)

6. configure ssh and gitconfig
https://github.com/bramus/freshinstall/blob/master/steps/2.ssh.sh

7. Install apps
```
brew install llvm cmake node graphviz boost hdf5 swig autojump ncdu tmux htop rg broot tldr wget
brew install --cask firefox google-chrome zoom nordvpn visual-studio-code telegram spectacle transmission  bitwarden vlc utm joplin cryptomator eul docker
```

-------------
Mac additional setting
- wallpaper, avatar, screen saver animation
- display larger text
- battery percentage (dock and menu bar -> battery)
- battery - turn off display after 1 hour
- remove unused items from menu bar with dragging to desktop
- notes view show folders
- ln -s ~/ ~/Desktop/alexyalunin
- true tone, night shift -> sunset to sunrise
- vscode -> file -> auto save
- vlc -> pref -> hotkeys -> faster fine ]
- [app-cleaner](https://github.com/sunknudsen/privacy-guides/blob/master/how-to-clean-uninstall-macos-apps-using-appcleaner-open-source-alternative/README.md)
