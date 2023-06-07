#include <stdio.h>
#include <stdlib.h>

int main() {

  int i = 100;
  int a = 200;
  char inputString[1000];

  gets(inputString);
  printf(inputString, i);

  return 0;
}

/*
 * Input: Hello World %d 
 * Output: Hello World 100 
 */
