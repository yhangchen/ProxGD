CC = g++
FLAGS = -std=c++11 -O2 -Wall

all: f.o Result.o ProxGD.o

.o: f.cpp ProxGD.h
g++ ProxGD.cpp h.cpp f.cpp Result.cpp line_search.cpp Test.cpp
Result.o: Result.cpp ProxGD.h
	$(CC) -o $@ Result.cpp $(FLAGS)
ProxGD.o: ProxGD.cpp ProxGD.h f.o Result.o
	$(CC) -o $@ ProxGD.cpp -lm f.o Result.o $(FLAGS)
