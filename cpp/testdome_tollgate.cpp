/*
   Question: TestDome - Tollgate
   Solution by Arash Partow 2014

   Implement a class Tollgate that will be used as a part of a larger
   simulation of crossing a bridge.

   The class Tollgate needs to contain the following methods:

   void enter(std::string licensePlate) - puts the specified car to the
   end of the line.

   std::string exit() - removes the first car from the line and returns
   its license plate. If no cars are in the line,
   std::logic_error("Tollgate is empty!") should be thrown.

   For example, the following code snippet should write "New York", "LA":
*/

#include <string>
#include <iostream>
#include <stdexcept>
#include <deque>

class Tollgate
{
public:

   void enter(std::string licensePlate)
   {
      l.push_back(licensePlate);
   }

   std::string exit()
   {
      if (l.size() == 0)
         throw std::logic_error("Tollgate is empty!");

      std::string r = l.front();

      l.pop_front();

      return r;
   }

private:

   std::deque<std::string> l;
};

#ifndef RunTests
int main(int argc, const char* argv[])
{
   Tollgate tollgate;
   tollgate.enter("New York");
   tollgate.enter("LA");
   std::cout << tollgate.exit() << '\n';
   std::cout << tollgate.exit() << '\n';
}
#endif