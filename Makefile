.PHONY: make run clean

k := 337

make:
	python test/data_unique.py
	python test/select_datapoint.py -n $(k)

run:
	python main.py -n $(k)
	python test/word_count_checker.py -n $(k)

clean:
	python test/clean.py -n $(k)