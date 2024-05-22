/*
   Question: TestDome - SearchStream
   Solution by Arash Partow 2014

   Implement the function findLines, which returns all lines from
   std::istream that contain the keyword.

   For example, for the keyword "there" and std::istream with the
   content:

   Hello, there!
   How are you today?
   Yes, you over there.

   The method should return the following lines:

   "Hello, there!"
   "Yes, you over there."
*/

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

class SearchStream
{
public:

   static std::vector<std::string> findLines(std::string keyword, std::istream& inStream)
   {
      std::vector<std::string> results;
      std::string buffer;

      while (std::getline(inStream,buffer))
      {
         if (buffer.find(keyword) != std::string::npos)
            results.push_back(buffer);
      }

      return results;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::string inMessage = "Hello, there!\nHow are you today?\nYes, you over there.";

   std::stringstream inStream(inMessage);
   std::vector<std::string> lines = SearchStream::findLines("there", inStream);

   for (size_t i = 0; i < lines.size(); i++)
      std::cout << lines.at(i) << "\n";

   return 0;
}
#endif