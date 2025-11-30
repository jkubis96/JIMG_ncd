.PHONY: format lint check all

format:
	isort jimgfl
	black jimgfl

lint:
	pylint --exit-zero --disable=import-error,no-member jimgfl

	

all: format lint
