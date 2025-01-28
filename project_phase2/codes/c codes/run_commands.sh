#!/bin/bash

# Run the first make command
make matrix_builder
if [ $? -ne 0 ]; then
  exit $?
fi

# Run the matrix_builder executable
./matrix_builder
if [ $? -ne 0 ]; then
  exit $?
fi

# Run the second make command
make
if [ $? -ne 0 ]; then
  exit $?
fi

# Run the test executable
./test
if [ $? -ne 0 ]; then
  exit $?
fi