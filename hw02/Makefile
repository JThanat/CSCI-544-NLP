split:
	@echo "Removing test_data"
	rm -rf ./test_data/negative_polarity
	rm -rf ./test_data/positive_polarity
	@echo "Removing negative_polarity"
	rm -rf ./train_data/negative_polarity
	@echo "Removing positive_polarity"
	rm -rf ./train_data/positive_polarity
	@echo "Splitting data..."
	python ./create_train_test.py 1

# 	@echo "Removing .DS_Store in test path"
# 	rm ./test_data/.DS_Store
train:
	python nblearn3.py ./train_data 10000

test:
	python nbclassify3.py ./test_data

validate:
	@echo "Validating data..."
	python validate.py
full:
	make train && make test && make validate
