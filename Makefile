
.PHONY: compile clean run

FLAGSMKL=-I"${MKLROOT}/include"
LINKMKL=-L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl



compile: program.x

clean:
	rm -f *.x

run: compile
	@echo "This fails:"   && ./program.x 1 0 || echo
	@echo "This runs ok:" && ./program.x 1 1 && echo
	@echo "This runs ok:" && ./program.x 1 2 && echo
	@echo "This runs ok:" && ./program.x 1 3 && echo
	@echo "This runs ok:" && ./program.x 0 0 && echo



program.x: source.cpp Makefile
	icpx -fsycl -g -O3 ${FLAGSMKL} $< -o $@ ${LINKMKL}
