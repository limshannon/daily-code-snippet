/*
   Question: TestDome - AreAnagrams
   Solution by Arash Partow 2014

   An anagram is a word formed from another by rearranging its letters,
   using all the original letters exactly once; for example, orchestra
   can be rearranged into carthorse.

   Write a function that checks if two words are each other's anagrams.

   For example, areStringsAnagrams("momdad", "dadmom") should return true
   as arguments are anagrams.
*/

#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>

class AreAnagrams
{
public:

   static bool areStringsAnagrams(std::string a, std::string b)
   {
      std::sort(a.begin(),a.end());
      std::sort(b.begin(),b.end());

      return (a == b);
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::cout << AreAnagrams::areStringsAnagrams("momdad", "dadmom");
   return 0;
}
#endif
