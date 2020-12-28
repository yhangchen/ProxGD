CC = g++
FLAGS = -std=c++11 -O2 -Wall

all: Test.o ProxGD.o

ProxGD.o: ProxGD.cpp h.cpp f.cpp Result.cpp line_search.cpp
    g++ ProxGD.cpp h.cpp f.cpp Result.cpp line_search.cpp
Test.o: Test.cpp ProxGD.o
	g++ Test.cpp ProxGD.o -o Test.o

clean:
	rm -f *.o