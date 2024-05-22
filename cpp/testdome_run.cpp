/*
   Question: TestDome - Run
   Solution by Arash Partow 2014

   Write a function that finds the zero-based index of the longest run in
   a string. A run is a consecutive sequence of the same character. If
   there is more than one run with the same length, return the index of
   the first one.

   For example, indexOfLongestRun("abbcccddddcccbba") should return 6 as
   the longest run is dddd and it first appears on index 6.
*/

#include <iostream>
#include <string>
#include <stdexcept>

class Run
{
public:

   static int indexOfLongestRun(std::string str)
   {
      int largest_run_length = 0;
      int largest_index      = 0;

      int  run_length        = 1;
      int  index             = 0;
      char curr_char         = str[0];

      for (std::size_t i = 1; i < str.size(); ++i)
      {
        if (str[i] == curr_char)
           ++run_length;
        else
        {
           if (run_length > largest_run_length)
           {
              largest_run_length = run_length;
              largest_index = index;
           }

           curr_char = str[i];
           run_length = 1;
           index = i;
        }
      }

      if (run_length > largest_run_length)
      {
         largest_run_length = run_length;
         largest_index = index;
      }

      return largest_index;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::cout << Run::indexOfLongestRun("abbcccddddcccbba");

   return 0;
}
#endif