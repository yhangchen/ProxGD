CC = g++
FLAGS = -std=c++11 -O2 -Wall

all: f.o Result.o ProxGD.o

f.o: f.cpp ProxGD.h
	$(CC) -o $@ f.cpp $(FLAGS)
Result.o: Result.cpp ProxGD.h
	$(CC) -o $@ Result.cpp $(FLAGS)
ProxGD.o: ProxGD.cpp ProxGD.h f.o Result.o
	$(CC) -o $@ ProxGD.cpp -lm f.o Result.o $(FLAGS)
