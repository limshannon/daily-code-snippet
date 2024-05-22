/*
   Question: TestDome - Common
   Solution by Arash Partow 2014

   Write a method that finds k-th most common element in a list.

   For example, kthMostCommon({ 5, 4, 3, 2, 1, 5, 4, 3, 2, 5, 4, 3, 5,
   4, 5 }, 2) should return 4. In the array, element 1 occurs once, 2
   twice, 3 three times, 4 four times, and 5 five times, making
   element 5 the most common element in the list and 4 the second most
   common element.
*/

#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <unordered_map>

class Common
{
public:

   static int kthMostCommon(const std::vector<int>& a, int k)
   {
      if (k > static_cast<int>(a.size()) || k <= 0)
         return -1;

      std::unordered_map<int,int> m;

      for (auto i : a)
      {
         m[i]++;
      }

      std::vector<std::pair<int,int>> n(m.size());

      std::copy(m.begin(),m.end(),n.begin());

      std::nth_element(n.begin(),n.begin() + (k - 1), n.end(),
                       [](auto x, auto y){ return y.second < x.second; });

      return (n.begin() + (k - 1))->first;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::vector<int> input;
   input.push_back(1);
   input.push_back(2);
   input.push_back(2);
   input.push_back(3);
   input.push_back(3);
   input.push_back(3);
   input.push_back(4);
   input.push_back(4);
   input.push_back(4);
   input.push_back(4);
   input.push_back(5);
   input.push_back(5);
   input.push_back(5);
   input.push_back(5);
   input.push_back(5);
   int x = Common::kthMostCommon(input, 2);
   std::cout << x;

   return 0;
}
#endif
