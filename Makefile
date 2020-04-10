NVCC := nvcc 
MAKE ?= make

LIBS := 

NVCCFLAGS := -I. \
			 -I./infra 

BUILT_LIBS := 

#LDFLAGS += $(shell pkg-config opencv --cflags --libs)

OBJS := *.o infra/*.o

all:
	$(MAKE) -C infra
	$(MAKE) app

app:
	$(NVCC) $(NVCCFLAGS) -c main.cu $(LDFLAGS)
	$(NVCC) $(OBJS) $(LIBS) $(BUILT_LIBS) -o run $(LDFLAGS)

clean:
	rm -f run
	rm -f *.o
	rm -f *.a

	$(MAKE) -C infra clean