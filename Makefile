.PHONY: make run clean note wc

k := 1
m := 20
make:
	pip install -r requirements.txt
	python test/data_unique.py
	python test/select_datapoint.py -n $(k)

run:
	python main.py -n $(k) -m $(m)
	# python test/word_count_checker.py -n $(k) -m $(m)

wc:
	python test/word_count_checker.py -n $(k) -m $(m)

clean:
	python test/clean.py -n $(k)

note: 
	python test/note.py

