/*
   Question: TestDome - Permissions
   Solution by Arash Partow 2014

   Write a function that converts a UNIX permission string into its
   numerical representation.

   The string consists of 3 blocks of 3 characters each. The returned
   number will have three digits, one for each block. Each digit is the
   sum of all permissions present in the block:

   CHARACTER   VALUE
   r           4
   w           2
   x           1
   -           0
*/

#include <stdexcept>
#include <iostream>
#include <string>

class Permissions
{
public:

   static int permStringToInt(std::string permString)
   {
      if (permString.size() != 9)
        return 0;

      int t      = 0;
      int cm     = 0;
      int result = 0;

      for (auto c : permString)
      {
         switch (c)
         {
            case 'r' : cm += 4; break;
            case 'w' : cm += 2; break;
            case 'x' : cm += 1; break;
         }

         t += 1;

         if ((t % 3) == 0)
         {
            result *= 10;
            result += cm;
            cm = 0;
         }
      }

      return result;

   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   // Should write 752
   std::cout << Permissions::permStringToInt("rwxr-x-w-");
   return 0;
}
#endif