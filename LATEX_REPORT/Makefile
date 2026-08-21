# Build the report. Requires TinyTeX on PATH (~/.local/bin).
MAIN = main

.PHONY: all clean watch

all:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
