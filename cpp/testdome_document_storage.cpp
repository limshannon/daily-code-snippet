/*
   Question: TestDome - DocumentStorage
   Solution by Arash Partow 2014

   A customer has complained that their application still occupies a big
   chunk of memory although they have removed all the documents from it.

   Fix the problem.
*/

#include <iostream>
#include <unordered_map>
#include <string>

class DocumentStorage
{
   class Document
   {
   public:

      Document(std::string title, std::string content)
      {
         this->title = title;
         this->content = content;
      }

      std::string title;
      std::string content;
   };

public:

   ~DocumentStorage()
    {
       clear();
    }

   void add(int id, std::string title, std::string content)
   {
      storage[id] = new Document(title, content);
   }

   void remove(int id, std::string& title, std::string& content)
   {
      std::unordered_map<int, Document*>::iterator it = storage.find(id);

      if (it == storage.end())
         return;

      Document* doc = it->second;
      title = doc->title;
      content = doc->content;

      delete doc;

      storage.erase(it);
   }

   void clear()
   {
      for (auto it : storage) delete it.second;
      storage.clear();
   }

private:

   std::unordered_map<int, Document*> storage;
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   DocumentStorage storage;
   storage.add(123456, "Hamlet", "Hamlet, Prince of Denmark.");
   storage.add(123457, "Othello", "Othello, the Moore of Venice.");

   std::string title, content;
   storage.remove(123456, title, content);

   std::cout << title << '\n';
   std::cout << content;

   storage.clear();

   return 0;
}
#endif
