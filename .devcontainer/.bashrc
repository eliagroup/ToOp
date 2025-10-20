# To enable GIT information on command prompt:
source /etc/bash_completion.d/git-prompt
# And also define PROMPT setup:
PS1='\[\e[1m\][\[\e[1;36m\]\u@docker\[\e[0m\]|\[\e[1;34m\]\w\[\e[0m\]\[\e[1m\]\[\033[00m\]\[\e[0;33m\]$(__git_ps1 " (git)-%s")\[\e[0m\]\[\e[1m\]] \[\e[0m\]'
export PS1

# Add a newline before each command:
PROMPT_COMMAND='echo'

# My 'ls' settings using coreutils-gls:
LS_COLORS=$LS_COLORS:'di=1;34:'
export LS_COLORS
alias ls='ls -h --color=auto --group-directories-first'
export QUOTING_STYLE=literal

# To make path output more readable:
function path() {
    old=$IFS
    IFS=:
    printf ${PATH//:/$'\n'}
    IFS=$old
}

# Copy pwd to the clipboard:
alias cpwd='printf "%q\n" "$(pwd)" | pbcopy && echo "Current directory copied to clipboard:" $(pbpaste)'

# Ensure venv is in PATH:
export PATH="/root/.venv/bin:$PATH"
export VIRTUAL_ENV="/root/.venv"
