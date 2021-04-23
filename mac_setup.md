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

4.  Install Git: 
```
brew install git
```

5. configure ssh and gitconfig
https://github.com/bramus/freshinstall/blob/master/steps/2.ssh.sh

6. Install apps
```
brew install tmux vlc skype zoom spectacle nordvpn ccleaner google-chrome visual-studio-code
```
App Store:

Internet:
- punto switcher 

7. install dotfiles from https://github.com/alexyalunin/dotfiles (use https://macos-defaults.com/ for custom settings)

-------------
Mac additional setting
- battery percentage
- turn off swipe back in trackpad - more gestures - swipe between pages
