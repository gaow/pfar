.PHONY: clean install docs

install:
	#@R CMD check ./ --no-manual -o $(shell mktemp -d tmp.XXXX)
	@R CMD build ./ --no-manual
	@((R CMD INSTALL pfar_*.tar.gz -l $(shell echo "cat(.libPaths()[1])" | R --slave) && rm -rf tmp.* pfar_*.tar.gz) || ($(ECHO) "Installation failed"))

docs:
	@echo 'roxygen2::roxygenise()' | R --vanilla --silent

clean:
	rm -f src/pfa.o src/pfar.so src/symbols.rds pfar_*.tar.gz
