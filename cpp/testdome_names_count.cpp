/*
   Question: TestDome - NamesCount
   Solution by Arash Partow 2014

   NamesCount is a component for analysing the proportion of names in a set of names.
*/

#include <iostream>
#include <string>
#include <map>

class NamesCount
{
public:

   NamesCount()
   {
      count = 0;
   }

   /**
   *   \brief Adds the name.
   *
   *   \param name Name.
   *
   **/
   void addName(std::string name)
   {
      counts[name] += 1;
      count++;
   }

   /**
   *   \brief Returns proportion of parameter name in all calls to addName.
   *
   *   \param name Name.
   *   \return double in interval [0, 1]. Returns 0 if addName has not been called.
   *
   **/
   double nameProportion(std::string name)
   {
      return (1.0 * counts[name]) / count;
   }

private:

   int count;
   std::map<std::string, int> counts;
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   NamesCount namesCount;

   namesCount.addName("James");
   namesCount.addName("John");
   namesCount.addName("Mary");
   namesCount.addName("Mary");

   std::cout << "Fraction of Johns: " << namesCount.nameProportion("John") << '\n';
   std::cout << "Fraction of Marys: " << namesCount.nameProportion("Mary") << '\n';

   return 0;
}
#endif
