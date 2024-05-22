/*
   Question: TestDome - AllAnagrams
   Solution by Arash Partow 2014

   An anagram is a word formed from another by rearranging its letters,
   using all the original letters exactly once; for example, orchestra
   can be rearranged into carthorse.

   Write a function which returns all anagrams of a given word (including
   the word itself) in any order.

   For example getAllAnagrams("abba") should return collection containing
   "aabb", "abab", "abba", "baab", "baba", "bbaa".
*/

#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>
#include <iterator>

class AllAnagrams
{
public:
   static std::vector<std::string> getAllAnagrams(std::string str)
   {

      std::unordered_set<std::string> anagram_set;

      std::sort(str.begin(), str.end());

      do
      {
         anagram_set.insert(str);
      }
      while(std::next_permutation(str.begin(), str.end()));

      std::vector<std::string> result;
      std::copy(anagram_set.begin(),anagram_set.end(),std::back_inserter(result));

      return result;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::vector<std::string> anagrams = AllAnagrams::getAllAnagrams("abba");

   for (std::vector<std::string>::iterator it = anagrams.begin(); it != anagrams.end(); ++it)
   {
      std::cout << *it << '\n';
   }

   return 0;
}
#endif
