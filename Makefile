#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

LIBINT2PATH=/usr/local/libint/2.2.0-alpha
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2


EXECUTABLE = w-embem.x

# change to icpc for Intel
CXX = g++
HOME = .
.SUFFIXES: .cc 


CFLAGS = -O2  -ffast-math    -march=native -std=c++11  $(LIBINT2INCLUDES)
LIBS = ./w-qcmol/libwqcmol.a -L$(LIBINT2PATH)/lib -lint2 -larmadillo -lblas -llapack 


SRC = main.cc pot_mbem.cc pot_embem.cc

COBJ=$(SRC:.cc=.o)

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

all	: $(EXECUTABLE) 


$(EXECUTABLE) : $(COBJ)
	$(CXX)   $(FLAGS) -o  $(EXECUTABLE) $(COBJ) $(LIBS) 

clean:
	rm *.o *~ *.x

# DO NOT DELETE
