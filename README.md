# Dust Project

> For dust thou art

In development

## Install

```bash
pip install -e .
```

## Train

```
python -m dust.init
python -m dust.train
```

## Continue training

```
python -m train --cont
```

## Demo

```
python -m --cont --demo --target_tick 500000
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
