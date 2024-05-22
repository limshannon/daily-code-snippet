/*
   Question: TestDome - ArrayMath
   Solution by Arash Partow 2014

   The code works correctly for the given example, but users have
   complained that it doesn't work for some of their cases.

   The method must throw std::invalid_argument("Argument exception") for
   every illegal argument value or combination. The expected arguments
   are two arrays with the same length.

   For example, weightedMean called with values {3, 6} and weights {4, 2}
   should return the weighted mean (3 * 4 + 6 * 2) / (4 + 2) = 4. Fix the
   bugs.
*/

#include <stdexcept>
#include <iostream>
#include <vector>

class ArrayMath
{
public:

   static double weightedMean(const std::vector<int>& values, const std::vector<int>& weights)
   {
      if (values.size() != weights.size())
         throw std::invalid_argument("Argument exception");

      double sum = 0.0;
      double sumWeights = 0.0;

      for (size_t i = 0; i < weights.size(); ++i)
      {
         sumWeights += weights[i];
      }

      for (size_t i = 0; i < values.size(); ++i)
      {
         sum += (values[i] * (1.0 * weights[i]));
      }

      return sum / sumWeights;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::vector<int> a, b;
   a.push_back(1);
   a.push_back(2);
   b.push_back(4);
   b.push_back(2);
   std::cout << ArrayMath::weightedMean(a, b);
   return 0;
}
#endif
