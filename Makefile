CC = nvcc
TARGETS = thrust singlethread multithread 
SRCS = thrust.cu singlethread.cu multithread.cu

all: $(TARGETS)

$(TARGETS): $(SRCS)
	@echo "Compiling $@ from $@.cu"
	$(CC) -o $@ $@.cu 

clean:
	rm -f $(TARGETS)
