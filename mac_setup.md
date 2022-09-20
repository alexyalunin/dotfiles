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
- split 4 window
- bar -> window -> save window arrangment 
- pref -> general -> startup -> open def window array
- profile -> terminal -> silence bell

4.  Install Git: 
```
brew install git
```

5. follow instructions on https://github.com/alexyalunin/dotfiles 
(use https://macos-defaults.com/ for custom settings, https://pawelgrzybek.com/change-macos-user-preferences-via-command-line/)

6. configure ssh and gitconfig
```
ssh-keygen -t rsa
```
create .extra and add line `ssh-add ~/.ssh/id_rsa`

add key to github with `cat ~/.ssh/id_rsa.pub`, then you can clone with `git clone git@...`

7. Install apps, but first put all apple apps into one folder
```
brew install llvm cmake node graphviz boost hdf5 swig autojump ncdu tmux htop rg broot tldr wget fzf
brew install --cask firefox google-chrome zoom visual-studio-code telegram spectacle vlc eul docker jetbrains-toolbox
brew install --cask transmission bitwarden cryptomator utm
```

-------------
Mac additional setting
- desktop -> wallpaper, screen saver animation (drift)
- users & groups -> avatar
- display -> resolution -> scaled -> larger text, check true tone, night shift -> sunset to sunrise
- dock and menu bar -> battery -> percentage
- battery -> turn off display after 15 mins
- touch id
- bluetooth -> show bluetooth in menu bar
- sound -> Uncheck the “Play user interface sound effects”
- keyboard -> shortcuts -> input sources -> cmd space (fix spotlight)

- notes: view -> show folders, edit -> substitutions -> smart links, preferences -> sort notes by date created
- install jetbrains apps, to use profile: bottom right - sync plugins silently, in preferences search font, set 12 and 0.9 
- firefox: default zoom firefox 90, pluggins (ublock, simple translate, cookies), https://www.google.com/preferences?hl=en check "Open each selected result in a new browser window", set firefox as default browser
- telegram: sounds -> sent message off, data and storage -> storage usage -> 5gb, preferences -> general -> show icon in menu bar off
- vscode: file -> auto save
- vlc: pref -> hotkeys -> faster fine ]

- remove items from menu bar with dragging to desktop (hold cmd)
- remove items from dock (drop to launchpad)
- [app-cleaner](https://github.com/sunknudsen/privacy-guides/blob/master/how-to-clean-uninstall-macos-apps-using-appcleaner-open-source-alternative/README.md)
