/*
   Question: TestDome - Palindrome
   Solution by Arash Partow 2014

   Write a function that checks if a given sentence is a palindrome. A
   palindrome is a word, phrase, verse, or sentence that reads the same
   backward or forward. Only the order of English alphabet letters (A-Z
   and a-z) should be considered, other characters should be ignored.

   For example, isPalindrome("Noel sees Leon.") should return true as
   spaces, period, and case should be ignored resulting with
   "noelseesleon" which is a palindrome since it reads same backward and
   forward.
*/

#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <algorithm>

class Palindrome
{
public:
   static bool isPalindrome(std::string str)
   {

      str.erase(std::remove_if(str.begin(),
                               str.end(),
                               [](char x) { return std::isalpha(x) == 0;}),
                               str.end());

      for (std::size_t i = 0; i < str.size() / 2; ++i)
      {
         auto c0 = std::tolower(str[i]);
         auto c1 = std::tolower(str[str.size() - i - 1]);

         if (c0 != c1)
            return false;
      }

      return true;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::cout << Palindrome::isPalindrome("Noel sees Leon.");
}
#endif
