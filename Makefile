.PHONY: clean install demo

install:
	#@R CMD check ./ --no-manual -o $(shell mktemp -d tmp.XXXX)
	@R CMD build ./ --no-manual
	@((R CMD INSTALL pfar_*.tar.gz -l $(shell echo "cat(.libPaths()[1])" | R --slave) && rm -rf tmp.* pfar_*.tar.gz) || ($(ECHO) "Please install the package manually with proper library path specified, e.g., R CMD INSTALL pfar_<version>.tar.gz -l /path/to/your/R/library/directory"))

demo:
	@echo 'library(pfar); dat = readRDS("vignettes/example_data.rds"); pfa(head(dat$$X), K = 15)' | R --vanilla --silent > tmp.txt

clean:
	rm -f src/pfa.o src/pfar.so src/symbols.rds pfar_*.tar.gz
