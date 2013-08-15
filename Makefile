CC:=gcc
LIB:=libopenclrt.so
EXAMPLE:=example/launcher

all: $(LIB) $(EXAMPLE)

$(LIB): opencl_runtime.c
	$(CC) -fPIC -shared opencl_runtime.c -o libopenclrt.so -I/usr/local/cuda/include -lOpenCL

$(EXAMPLE):example/launcher.c
	$(CC) -o example/launcher example/launcher.c -L./ -lopenclrt -I/usr/local/cuda/include -I. -lOpenCL
