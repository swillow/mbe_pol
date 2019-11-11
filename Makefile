#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib
# libint2 directory
LIBINT2PATH=/home/willow/prog
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2


EXECUTABLE = mbe_pol.x

CXX = mpic++
HOME = .
.SUFFIXES: .cc 


CFLAGS = -O2  -ffast-math    -march=native -std=c++11  \
	 -I/usr/include/eigen3 $(LIBINT2INCLUDES)

LIBS = ./w-qcmol/libwqcmol.a -L$(LIBINT2PATH)/lib -lint2 -larmadillo 


SRC = message.cc pes.cc main.cc

COBJ=$(SRC:.cc=.o)

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

all	: $(EXECUTABLE) 


$(EXECUTABLE) : $(COBJ)
	$(CXX)   $(FLAGS) -o  $(EXECUTABLE) $(COBJ) $(LIBS) 

clean:
	rm *.o *~ *.x

# DO NOT DELETE
