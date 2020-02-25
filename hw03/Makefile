split:
	@echo "Removing test_data"
	rm -rf ./test_data/negative_polarity
	rm -rf ./test_data/positive_polarity
	@echo "Removing negative_polarity"
	rm -rf ./train_data/negative_polarity
	@echo "Removing positive_polarity"
	rm -rf ./train_data/positive_polarity
	@echo "Splitting data..."
	python ./create_train_test.py

# 	@echo "Removing .DS_Store in test path"
# 	rm ./test_data/.DS_Store
train:
	python perceplearn3.py ./train_data

testVani:
	python percepclassify3.py ./vanillamodel.txt ./test_data

testAvg:
	python percepclassify3.py ./averagedmodel.txt ./test_data

validate:
	@echo "Validating data..."
	python validate.py

full:
	make train && make testVani && make validate && make testAvg && make validate

fullVani:
	make train && make testVani && make validate

fullAvg:
	make train && make testAvg && make validate