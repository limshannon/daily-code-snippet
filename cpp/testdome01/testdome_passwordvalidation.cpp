/*
   Question: TestDome - PasswordValidation
   Solution by Arash Partow 2014

   Implement the strongPassword function, which accepts a password and
   returns true if the password is strong and false if it's not.

   A password is strong if it satisfies the following rules:

   It is at least 12 characters long.

   It contains at least one uppercase letter, at least one lowercase
   letter and at least one digit.

   Letters should be split by digits into at least two groups (like
   "Strong1Password").

   For example, "strong1password" is not a strong password because it
   does not contain any uppercase letters.
*/

#include <string>
#include <cctype>

class PasswordValidation
{
public:

   static bool strongPassword(std::string password)
   {
      if (password.size() < 12)
         return false;

      std::size_t ucc = 0;
      std::size_t lcc = 0;
      std::size_t dc  = 0;

      for (auto i : password)
      {
         if ('a' <= i && i <= 'z')
            ++lcc;
         else if ('A' <= i && i <= 'Z')
            ++ucc;
         else if (std::isdigit(i))
            ++dc;
      }

      if (ucc < 1 || lcc < 1 || dc < 1)
         return false;

      std::size_t x = 0;
      while (x < password.size() && std::isdigit(password[x])) ++x;

      std::size_t y = password.size() - 1;
      while (y > 0 && std::isdigit(password[y])) y -= 1;

      if (x >= y)
         return false;

      return ((((y - x) + 1) - (ucc + lcc))) > 0;
   }
};

