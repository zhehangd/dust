# Dust Project

> For dust thou art

In development


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
