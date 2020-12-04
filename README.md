# Dust Project

> For dust thou art

In development

## Install

```bash
pip install -e .
```

## Train

```
python -m dust.init --proj_name testproj
cd testproj
python -m dust.train
```

## Unit Test

```
pytest
```

## Doc

```bash
cd docs
sphinx-apidoc -f -o ./source/ ../dust
make html
```
