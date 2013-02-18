CC = gcc
CPP = g++

BIN = ./test
C_SRCS = kdtree.c 
CPP_SRCS = pomdp.cpp test.cpp sarsop.cpp

C_OBJS = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(C_OBJS) $(CPP_OBJS)

INCS = -I /home/pratik/code/utils -I /home/pratik/apps
LIBS = 

CFLAGS = -Wall -g -pg
LDFLAGS =

all: $(BIN)

$(BIN): $(OBJS)
	@echo "  [LD]    $@"
	@$(CPP) $(LDFLAGS) -o $(BIN) $(OBJS) $(LIBS)

%.o: %.c
	@echo "  [CPP]   $@"
	@$(CC) $(CFLAGS) $(INCS) -c $< -o $@

%.o: %.cpp
	@echo "  [CC]    $@"
	@$(CPP) $(CFLAGS) $(INCS) -c $< -o $@

clean:
	rm -rf $(BIN) $(C_OBJS) $(CPP_OBJS) gmon.out

ctags:
	ctags -R $(C_SRCS) $(CPP_SRCS) *.h

depend:
	makedepend -f deps $(INCS) -- $(CFLAGS) -- $(C_SRCS) -- $(CPP_SRCS)

docs:
	doxygen Doxyfile
include deps
