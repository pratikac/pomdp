CC = gcc
CPP = g++

BIN = ./test
C_SRCS = kdtree.c 
CPP_SRCS = logger.cpp mymath.cpp sarsop.cpp test.cpp

C_OBJS = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(C_OBJS) $(CPP_OBJS)

INCS =
LIBS = 

CFLAGS = -Wall -g -pg -O3 
LDFLAGS = -g -pg 

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
	makedepend -f deps -- $(CFLAGS) -- $(C_SRCS) -- $(CPP_SRCS)

include deps
