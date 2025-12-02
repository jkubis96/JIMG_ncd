.PHONY: format lint all

format:
	isort jimg_ncd
	black jimg_ncd
	isort tests
	black tests

lint:
	pylint --exit-zero --disable=import-error,no-member jimg_ncd

	

all: format lint
