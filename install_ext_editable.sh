check_editable() {
    package="$1"

    if pip show $package | grep -qe '^Editable project location'; then
        return 0
    else
        return 1
    fi
}

if check_editable e3nn; then
    echo e3nn already installed in editable mode
else
    echo 'e3nn not editable; installing'
    pip install -e ./external_libs/e3nn
fi
if check_editable sevenn; then
    echo sevenn already installed in editable mode
else
    echo 'sevenn not editable; installing'
    pip install -e ./external_libs/SevenNet
fi
if check_editable superionic_toy; then
    echo superionic_toy already installed in editable mode
else
    echo 'superionic_toy not editable; installing'
    pip install -e ./external_libs/SuperionicToyMD
fi