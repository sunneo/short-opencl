CC:=gcc
EXAMPLE:=launcher example sizeT VectorAdd
LDFLAGS+=-L./  -lopenclrt -lOpenCL 
CFLAGS+=  -I../ -I/usr/local/cuda/include

all: $(EXAMPLE) 

example: example1.c
	$(CC) -o example example1.c $(LDFLAGS) $(CFLAGS)

launcher: launcher.c
	$(CC) -o launcher launcher.c $(LDFLAGS) $(CFLAGS)

VectorAdd: VectorAdd.c
	$(CC) -o VectorAdd VectorAdd.c $(LDFLAGS) $(CFLAGS)

sizeT: sizeT.c
	$(CC) -o sizeT sizeT.c $(LDFLAGS) $(CFLAGS)


clean:
	rm -fv $(EXAMPLE)

