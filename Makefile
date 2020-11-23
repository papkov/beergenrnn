TEST_PATH=./tests
MODULES_PATH=./beergenrnn
RESULTS_PATH=./results
TRAIN_PATH=main.py

.PHONY: format
format:
	isort $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

.PHONY: lint
lint:
	isort -c $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	black --check $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)
	mypy $(MODULES_PATH) $(TRAIN_PATH) $(TEST_PATH)

.PHONY: test
test:
	python -m unittest discover -s $(TEST_PATH) -t $(TEST_PATH)