/*
   Question: TestDome - Path
   Solution by Arash Partow 2014

   Write a function that provides change directory (cd) function for an
   abstract file system.

   Notes:

   1. Root path is '/'.
   2. Path separator is '/'.
   3. Parent directory is addressable as "..".
   4. Directory names consist only of English alphabet letters (A-Z and a-z).

   For example, Path("/a/b/c/d").cd("../x").getPath() should return "/a/b/c/x".

   Note: Do not use built-in path-related functions.
*/

#include <iostream>
#include <string>
#include <stdexcept>

class Path
{
public:

   Path(std::string path)
   {
      currentPath = path;
   }

   std::string getPath() const
   {
      return currentPath;
   }

   Path cd(std::string newPath) const
   {
      Path p = *this;

      if (newPath == "/")
      {
         p.currentPath = "/";

         return p;
      }

      while (newPath.size())
      {
         if (newPath.size() >= 2)
         {
            if (newPath[0] == '.' && newPath[1] == '.')
            {
               if (!p.currentPath.empty())
               {
                  p.currentPath.erase(p.currentPath.find_last_of("/\\"));

                  if (p.currentPath.empty())
                  {
                     p.currentPath = "/";
                  }
               }

               newPath.erase(0, 2);

               continue;
            }
         }

         if (newPath[0] == '/' || newPath[0] == '\\')
         {
            newPath.erase(0, 1);

            if (newPath[0] == '.')
               continue;
         }

         auto i = newPath.find_first_of("/\\");

         if (p.currentPath.back() != '/' && p.currentPath.back() != '\\')
         {
            p.currentPath += "/";
         }

         p.currentPath += newPath.substr(0, i);

         newPath.erase(0, i);
      }

      return p;
   }

private:

   std::string currentPath;
};


#ifndef RunTests
int main()
{
    Path path("/a/b/c/d");
    std::cout << path.cd("../x").getPath();
}
#endif
