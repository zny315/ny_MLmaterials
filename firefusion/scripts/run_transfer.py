#!/usr/bin/env python3
from firefusion.cli import main

if __name__ == "__main__":
    main(["transfer", *__import__("sys").argv[1:]])
