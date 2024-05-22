/*
   Question: TestDome - CachingCalculator
   Solution by Arash Partow 2014

   The caching feature does not work in the calculator implementation provided.

   Find and fix the bug.
*/

#include <iostream>
#include <unordered_map>

class CachingCalculator
{
   class Arguments
   {
   public:

      Arguments(int firstArgument, int secondArgument)
      {
         this->firstArgument = firstArgument;
         this->secondArgument = secondArgument;
      }

      int firstArgument;
      int secondArgument;
   };

   struct KH
   {
      std::size_t operator()(const Arguments& a) const
      {
         return  std::hash<int>()(a.firstArgument) ^
                (std::hash<int>()(a.secondArgument) << 1);
      }
   };

   struct KE
   {
      bool operator()(const Arguments& lhs, const Arguments& rhs) const
      {
         return lhs.firstArgument  == rhs.firstArgument  &&
                lhs.secondArgument == rhs.secondArgument ;
      }
   };

public:

   CachingCalculator(int(*func)(int, int))
   {
      this->func = func;
   }

   int calculate(int firstArgument, int secondArgument)
   {
      Arguments args(firstArgument, secondArgument);

      std::unordered_map<Arguments, int, KH, KE>::iterator it = calculations.find(args);

      if (it != calculations.end())
         return it->second;

      int calculation = func(firstArgument, secondArgument);

      calculations[args] = calculation;

      return calculation;
   }

private:

   std::unordered_map<Arguments, int, KH, KE> calculations;

   int(*func)(int, int);
};


#ifndef RunTests
int modulo(int a, int b)
{
   std::cout << "Function modulo has been called.\n";
   return a % b;
}

int main(int argc, const char* argv[])
{
   CachingCalculator calculator(modulo);

   // Function modulo should be called.
   std::cout << calculator.calculate(5, 2) << '\n';

   // Function modulo should be called.
   std::cout << calculator.calculate(7, 4) << '\n';

   // Function modulo shouldn't be called because we have already made a call with arguments (5, 2)!
   // Instead, result should be pulled from the cache!
   std::cout << calculator.calculate(5, 2) << '\n';

   return 0;
}
#endif