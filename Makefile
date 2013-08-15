CC:=gcc
LIB:=libopenclrt.so
EXAMPLE:=launcher

all: $(LIB) $(EXAMPLE)

$(LIB): opencl_runtime.c
	$(CC) -fPIC -shared opencl_runtime.c -o libopenclrt.so -I/usr/local/cuda/include -lOpenCL

$(EXAMPLE):launcher.c
	$(CC) -o launcher launcher.c -L./ -lopenclrt -I/usr/local/cuda/include -lOpenCL
