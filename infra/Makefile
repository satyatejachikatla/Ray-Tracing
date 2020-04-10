NVCC := nvcc
MAKE ?= make

LIBS := 

NVCCFLAGS := -I. \
			 -I.. 

#LDFLAGS += $(shell pkg-config opencv --cflags --libs)

OBJS := *.o

all:
	$(NVCC) $(NVCCFLAGS) -c *.cu $(LDFLAGS)
clean:
	rm -f *.o
	rm -f *.a