trainIt:
	python hmmlearn3.py ./hmm-training-data/it_isdt_train_tagged.txt

decodeIt:
	python hmmdecode3.py ./hmm-training-data/it_isdt_dev_raw.txt

valIt:
	python validate.py ./hmmoutput.txt ./hmm-training-data/it_isdt_dev_tagged.txt

trainJap:
	python hmmlearn3.py ./hmm-training-data/ja_gsd_train_tagged.txt

decodeJap:
	python hmmdecode3.py ./hmm-training-data/ja_gsd_dev_raw.txt

valJap:
	python validate.py ./hmmoutput.txt ./hmm-training-data/ja_gsd_dev_tagged.txt

fullIt:
	make trainIt && make decodeIt && make valIt

fullJap:
	make trainJap && make decodeJap && make valJap

full:
	@echo "Validating Italian..."
	make fullIt
	@echo "Validating Jap..."
	make fullJap
