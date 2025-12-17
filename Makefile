init:
	python3 -m pip install -r requirements.txt

test:
	pytest --cov=bitcoin_scalper tests/

lint:
	flake8 bitcoin_scalper/ tests/

docs:
	cd docs && make html

train:
	python3 train.py

.PHONY: init test lint docs train
