# TinyTitle runtime
CC ?= cc
CFLAGS ?= -O2 -march=native -Wall

all: title-v1

title-v1: main.c ttm.h
	$(CC) $(CFLAGS) -o $@ main.c -lm

# sanitizer build for the adversarial gate
asan: main.c ttm.h
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o title-v1-asan main.c -lm

musl: main.c ttm.h
	musl-gcc -O2 -static -o title-v1-musl main.c -lm

clean:
	rm -f title-v1 title-v1-asan title-v1-musl

.PHONY: all asan musl clean
