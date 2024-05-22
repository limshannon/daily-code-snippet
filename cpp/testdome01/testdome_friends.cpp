/*
   Question: TestDome - Friends
   Solution by Arash Partow 2014

   Given a data structure representing a social network, write a function
   that finds friends of a certain degree. Friends of the first degree
   are a member's immediate friends, friends of the second degree are
   friends of a member's friends excluding first degree friends, etc.

   For example, if A is a friend with B and B is a friend with C, then
   getFriendsOfDegree(A, 2) should return C since C is the only second
   degree friend of A (B is a first degree friend of A).
*/

#include <string>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <deque>


class Member
{
public:

   Member(std::string email)
   {
      this->email = email;
   }

   std::string getEmail() const
   {
      return this->email;
   }

   std::vector<Member*> getFriends() const
   {
      return this->friends;
   }

   void addFriend(Member* _friend)
   {
      this->friends.push_back(_friend);
   }

private:

   std::string email;
   std::vector<Member*> friends;
};

class Friends
{
public:

   static std::vector<Member*> getFriendsOfDegree(Member* member, int degree)
   {
      std::deque<Member*> p;

      p.push_back(member);

      std::unordered_set<Member*> seen_set;

      while (degree > 1)
      {
         if (p.empty())
           break;

         std::deque<Member*> p2;

         for (auto v : p)
         {
            seen_set.insert(v);

            std::vector<Member*> f = v->getFriends();

            for (auto ff : f)
            {
               if (seen_set.count(ff))
                  continue;

               p2.push_back(ff);
            }
         }

         p = p2;
         --degree;
      }

      if (degree == 1)
      {
         std::vector<Member*> x;

         for (auto v : p)
         {
            seen_set.insert(v);
         }

         for (auto v : p)
         {
            std::vector<Member*> f = v->getFriends();

            for (auto ff : f)
            {
               if (seen_set.count(ff))
                  continue;

               seen_set.insert(ff);

               x.push_back(ff);
            }
         }

         return x;
      }
      else
         return std::vector<Member*>();

   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   Member a("A");
   Member b("B");
   Member c("C");

   a.addFriend(&b);
   b.addFriend(&c);

   std::vector<Member*> friends = Friends::getFriendsOfDegree(&a, 2);
   for (std::vector<Member*>::iterator it = friends.begin(); it != friends.end(); ++it)
   {
      std::cout << (*it)->getEmail() << '\n';
   }
}
#endif
