# GAT setup

## Install

```bash
poetry install
```

## Run

```bash
poetry run python run.py
```

## Format

```bash
poetry run ruff check . --fix

# nix version
ruff check . --fix
```

## Nix

```nix
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell rec {

  buildInputs = [
    pkgs.python3
    pkgs.poetry
    pkgs.zlib
    pkgs.ruff
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}
```
