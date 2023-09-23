#!/bin/bash

# This script sets up your local development environment.
# Currently, the script only support MacOS.
# Something not working? Submit a PR!
# Order of operations matters

# Fail on errors or unassigned variables.
set -eu

function echo_step() {
    GREEN='\033[1;32m'
    NC='\033[0m'
    MSG="$1"

    echo -e "\n${GREEN}* ${MSG}${NC}"
}

function echo_warn() {
    YELLOW='\033[1;33m'
    NC='\033[0m'
    MSG="$1"

    echo -e "\n${YELLOW}* ${MSG}${NC}"
}

function add_to_shell_profile() {
    mkdir -p "$(dirname "$1")"
    touch "$1"
    if [ -f "$1" ] && ! grep -q direnv "$1" 2> /dev/null; then
        echo "$2" >> "$1"
    fi
}

########### OS #################
function check_os() {
    echo_step "Checking OS.."

    # APPLE_CPU indicates if the machine is using a M1, M2 etc.
    APPLE_CPU=false
    case "$(uname -s)" in
        Darwin*)
            MACHINE=darwin
            # Check cpu brand. We can't use arch here because we're using rosetta.
            [[ $(sysctl -n machdep.cpu.brand_string) =~ "Apple" ]] && APPLE_CPU=true || APPLE_CPU=false
            ;;
        Linux*)
            bash ./bin/setup_linux.sh
            exit
            ;;
        *)
            echo "This script only works on MacOS. Found unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac
}

########### Homebrew ###########
function set_homebrew() {
    echo_step "Setting up homebrew..."
    BREW_INSTALLATION_SCRIPT_URL="https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"

    case $MACHINE in
        darwin)
            if ! which brew >/dev/null 2>&1; then
                yes | /bin/bash -c "$(curl -fsSL $BREW_INSTALLATION_SCRIPT_URL)"
            fi

            brew update
            brew -v
    esac
}

############ Python ############
function setup_python() {
    echo_step "Setting up Python..."

    if [ -z ${PYENV_VERSION+x} ]; then
        export PYENV_VERSION=3.10.6;
    fi

    if [ ! "$(python --version)" == "Python ${PYENV_VERSION}" ]; then
        echo_warn "Expected Python version ${PYENV_VERSION} not found. Installing..."
        pyenv install "${PYENV_VERSION}"
        pyenv global "$PYENV_VERSION"
    fi

    if [ ! "$(~/.pyenv/versions/${PYENV_VERSION}/bin/python --version)" == "Python ${PYENV_VERSION}" ]; then
        echo_warn "Failed to set up Python ${PYENV_VERSION} environment."
        exit 1
    else
        echo "Python ${PYENV_VERSION} setup complete."
    fi
}

############# Direnv #############
function setup_direnv() {
    echo_step "Setting up direnv..."

    brew install direnv
    direnv allow

    case $(basename "$SHELL") in
        bash)
            if [ ! -f ~/.bash_profile ]; then
                touch ~/.bash_profile
            fi

            LOAD_BASHRC="if [  -f $HOME/.bashrc ]; then source $HOME/.bashrc; fi"

            # Load bashrc from bash_profile if not already loaded.
            if ! grep -q bashrc ~/.bash_profile 2> /dev/null; then
                echo "$LOAD_BASHRC" >> ~/.bash_profile
            fi

            add_to_shell_profile ~/.bashrc 'eval"$(direnv hook bash)"'
        ;;
        zsh)
            add_to_shell_profile ~/.zshrc 'eval "$(direnv hook zsh)"'
        ;;
        fish)
            add_to_shell_profile ~/.config/fish/config.fish 'eval "$(direnv hook fish)"'
        ;;
        *)
            echo_warn "Unsupported shell for direnv. Only bash, zsh, and fish supported."
            exit 1
    esac

    direnv export $(basename "$SHELL")
}

########## Poetry #############
function setup_pip() {
    echo_step "Upgrading pip..."
    python -m pip install --upgrade pip
}

########## poetry dependencies ############
function install_deps() {
    echo_step "Installing dependencies..."
    python -m pip install -e ."[dev]"
}

######## setup pyenv ############
function setup_pyenv() {
    echo_step "Setting up Pyenv"

    pyenv_init_shim="$(pyenv init --path)"
    shell_profile=~/.zshrc
    case $(basename "$SHELL") in
        bash)
            shell_profile=~/.bashrc
        ;;
        zsh)
            shell_profile=~/.zshrc
        ;;
        fish)
            shell_profile=~/.config/fish/config.fish
        ;;
        *)
            echo_warn "Unsupported shell for pyenv. Only bash, zsh, and fish supported."
            exit 1
    esac

    if ! command -v pyenv &> /dev/null
    then
        echo "pyenv not found, installing..."
        brew update && brew install pyenv
        add_to_shell_profile $shell_profile 'eval "$(pyenv init --path)"'
        add_to_shell_profile $shell_profile 'eval "$(pyenv init -)"'
    else
        echo "pyenv found, upgrading to latest version..."
        brew update && brew upgrade pyenv
    fi
}

######### skip steps logic ############
skip_steps=''
while getopts "s:" opt; do
  case $opt in
    s)
      echo_step "-s was triggered, Skipping: $OPTARG"
      skip_steps="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))


function skip_or_run() {
    if [[ "$skip_steps" != *"$2"* ]]; then
        $1
    fi
}

########## reloading shell helper ###########
function reload_shell() {
    case $(basename "$SHELL") in
        bash)
            source ~/.bashrc
        ;;
        zsh)
            source ~/.zshrc # FIXME this triggers an error getting run from a bash shell
        ;;
        fish)
            source ~/.config/fish/config.fish
        ;;
        *)
            echo "Unsupported shell. Only bash, zsh, and fish supported."
            exit 1
    esac
}

############ execute setup steps ##############
echo_warn 'You can skip steps by passing [-s "os brew python direnv poetry deps"]'
skip_or_run check_os "os"
skip_or_run set_homebrew "brew"
skip_or_run setup_pyenv "pyenv"
skip_or_run setup_python "python"
skip_or_run setup_direnv "direnv"
skip_or_run setup_pip "pip"
skip_or_run install_deps "deps"
echo_warn "Congratulations - your local setup is complete!\n\nPlease reload your shell..."
# reload_shell