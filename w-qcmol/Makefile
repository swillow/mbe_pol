#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

# libint2 directory
LIBINT2PATH=/home/willow/prog/libint2
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

EXECUTABLE = qcmol.x

# change to icpc for Intel
CXX = clang++
HOME = .
.SUFFIXES: .cc 

CFLAGS = -O3 -fPIC -std=c++11  \
	   -I/usr/include/eigen3 $(LIBINT2INCLUDES)

#CFLAGS = -g   -march=native -std=c++11 $(LIBINT2INCLUDES)  
LIBS =  -L$(LIBINT2PATH)/lib -lint2  -larmadillo 

SRC = Integrals.cc RHF.cc MP2.cc MP2ints.cc MP2cphf.cc esp.cc 

QCMOL_OBJ=$(SRC:.cc=.o)

.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

all	: libwqcmol.a $(EXECUTABLE) 


libwqcmol.a : $(QCMOL_OBJ)
	$(AR) cr libwqcmol.a $(QCMOL_OBJ)


$(EXECUTABLE) : main.o libwqcmol.a
	$(CXX)   $(FLAGS) -o  $(EXECUTABLE) main.o libwqcmol.a $(LIBS) 


clean:
	rm *.o libwqcmol.a qcmol.x

# DO NOT DELETE
