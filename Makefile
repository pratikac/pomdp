CC = gcc
CPP = g++

BIN = ./main
C_SRCS = utils/kdtree.c
CPP_SRCS = utils/common.cpp systems/singleint.cpp mdp.cpp main.cpp

C_OBJS = $(C_SRCS:.c=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(C_OBJS) $(CPP_OBJS)

INCS = -I./ -I systems/ -I utils/ -I /home/pratik/apps/eigen
LIBS = 

CFLAGS = -Wall -g -pg -O0 `pkg-config --cflags bot2-core`
LDFLAGS = -g -pg `pkg-config --libs bot2-lcmgl-client` \
		  `pkg-config --libs bot2-core`

#@touch data/monte_carlo.dat data/rrg.dat data/rrgp.dat data/traj.dat
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
