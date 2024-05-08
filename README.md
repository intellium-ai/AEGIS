# AEGIS
Autonomous Embedded-Graph Intrusion Sentinel


## Install

This can be a bit tricky, follow these steps exactly

1. Install poetry as usual

```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Add the following line to your `.bashrc` or `.zshrc`

```
export PATH="$HOME/.local/bin:$PATH"
```

3. You can verify the installation has worked by reloading your shell and running

```
poetry --version
```

4. Create your poetry environment.

```
poetry env use $(which python3)
poetry shell
```

Verify python and pip are coming from your virtual environment by running `which python` and `which pip`. The path should look something like:

```
/home/$USER/.cache/pypoetry/virtualenvs/aegis-[something]-py3.10/bin/[python|pip]
```

4. Run the following commands precisely. Do not miss anything. Make sure your poetry shell is activated.

```
pip install --upgrade pip==23.0.1
pip install wheel==0.38.4 --upgrade
pip install setuptools==66 --upgrade
pip install --no-use-pep517 gym==0.21.0
```

5. You can now install the project with

```
poetry install
```


