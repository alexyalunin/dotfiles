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
brew install vlc zoom spectacle nordvpn ccleaner google-chrome visual-studio-code telegram folx cleanmymac 1password 
brew install tmux htop rg broot tldr wget
```
App Store:

Internet:
- punto switcher 

-------------
Mac additional setting
- wallpaper, avatar, screen saver animation
- display larger text
- battery percentage (dock and menu bar -> battery)
- battery - turn off display after 1 hour
- remove unused items from menu bar with dragging to desktop
- notes view show folders
- cd Downloads && ln -s . ~/Desktop/Downloads
- true tone, night shift -> sunset to sunrise