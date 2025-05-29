init:
	python3 -m pip install -r requirements.txt

test:
	pytest --cov=bitcoin_scalper tests/

lint:
	flake8 bitcoin_scalper/ tests/

docs:
	cd docs && make html

.PHONY: init test lint docs
