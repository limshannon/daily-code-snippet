/*
   Question: TestDome - Frog
   Solution by Arash Partow 2014

   A frog only moves forward, but it can move in steps 1 inch long or in
   jumps 2 inches long. A frog can cover the same distance using
   different combinations of steps and jumps.

   Write a function that calculates the number of different combinations
   a frog can use to cover a given distance.

   For example, a distance of 3 inches can be covered in three ways:
   step-step-step, step-jump, and jump-step.
*/

#include <iostream>
#include <stdexcept>
#include <unordered_map>

class Frog
{
public:

   static int numberOfWays(int n)
   {
      if (n == 1)
        return 1;
      else if (n == 2)
        return 2;
      else
      {
        static std::unordered_map<int,int> m;

        auto i = m.find(n);

        if (i != m.end())
           return i->second;

        int t = numberOfWays(n - 1) + numberOfWays(n - 2);

        m[n] = t;

        return t;
      }
   }

private:
};

#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::cout << Frog::numberOfWays(3);

   return 0;
}
#endif
