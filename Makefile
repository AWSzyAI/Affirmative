.PHONY: make run clean

k := 10

make:
	python test/data_unique.py
	python test/select_datapoint.py -n $(k)

run:
	python main.py -k $(k)
	python test/word_count_checker.py

clean:
	python test/clean.py