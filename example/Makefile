CC:=gcc
EXAMPLE:=launcher example1
LDFLAGS+=-L./  -lopenclrt -lOpenCL
CFLAGS+=  -I../ 

all: $(EXAMPLE) 

example1: example1.c
	$(CC) -o example example1.c $(LDFLAGS) $(CFLAGS)

clean:
	rm -fv $(EXAMPLE)

