CC = gcc-4.7
CPP = g++-4.7

BIN = ./test
C_SRCS = utils/kdtree.c 
CPP_SRCS = utils/utils.cpp pomdp/model.cpp test.cpp

C_OBJS = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(C_OBJS) $(CPP_OBJS)

INCS = -I ./ -I /home/pratik/apps -I /home/pratik/apps/Eigen/unsupported -I ./utils/ -I ./solver/ -I ./pomdp/ -I ./system/
LIBS = -L parser/ -lparser -lfl

CFLAGS = -Wall -pg -g -O3 -ggdb
CPPFLAGS = -Wall -pg -g -O3 -ggdb -std=c++11
LDFLAGS = -pg

all: $(BIN)

$(BIN): $(OBJS)
	@echo "  [LD]    $@"
	@$(CPP) $(LDFLAGS) -o $(BIN) $(OBJS) $(LIBS)

%.o: %.c
	@echo "  [CC]     $@"
	@$(CC) $(CFLAGS) $(INCS) -c $< -o $@

%.o: %.cpp
	@echo "  [CPP]    $@"
	@$(CPP) $(CPPFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(BIN) $(C_OBJS) $(CPP_OBJS)

ctags:
	ctags -R $(C_SRCS) $(CPP_SRCS) *.h

depend:
	makedepend -f deps $(INCS) -- $(CFLAGS) -- $(C_SRCS) -- $(CPP_SRCS) -Y

docs:
	doxygen Doxyfile

include deps
# DO NOT DELETE
