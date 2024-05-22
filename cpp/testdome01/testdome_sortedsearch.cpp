/*
   Question: TestDome - SortedSearch
   Solution by Arash Partow 2014

   Implement function countNumbers that accepts a sorted vector of
   integers and counts the number of vector elements that are less than
   the parameter lessThan.

   For example, for vector v containing { 1, 3, 5, 7 },
   SortedSearch::CountNumbers(v, 4) should return 2 because there are two
   vector elements less than 4.
*/

#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

class SortedSearch
{
public:

   static int countNumbers(const std::vector<int>& sortedVector, int lessThan)
   {
      return std::distance
                  (
                    sortedVector.begin(),
                    std::lower_bound(sortedVector.begin(), sortedVector.end(), lessThan)
                  );
   }
};

#ifndef RunTests
int main()
{
   std::vector<int> v { 1, 3, 5, 7 };
   std::cout << SortedSearch::countNumbers(v, 4);
}
#endif
