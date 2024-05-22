/*
   Question: TestDome - Chain
   Solution by Arash Partow 2014

   You are holding one link of a chain in your hand. Implement function
   whichSideIsLonger to find which side of the chain, relative to the
   link you are holding, has more links.

   If the left side has more links return Side::LEFT, if the right side
   has more links return Side::RIGHT, and if both sides have an equal
   number of links or if the chain is a closed loop, return Side::NONE.
   For example, for the code below, the output should be Side::RIGHT (or
   2):

   Link* left = new Link();
   Link* middle = new Link();
   Link* right = new Link();
   left->append(middle);
   middle->append(right);
   std::cout << Chain::whichSideIsLonger(*left);
*/

#include <stdexcept>
#include <iostream>
#include <unordered_set>

enum Side { NONE, LEFT, RIGHT };

class Link
{
public:

   Link* getLeft() const
   {
      return this->left;
   }

   Link* getRight() const
   {
      return this->right;
   }

   void append(Link* rightPart)
   {
      if (this->right != NULL)
         throw std::logic_error("Link is already connected.");

      this->right = rightPart;
      rightPart->left = this;
   }

private:

   Link* left;
   Link* right;
};

class Chain
{
public:

   static Side whichSideIsLonger(const Link& link)
   {
      int l = 0;
      int r = 0;

      std::unordered_set<const Link*> s;

      {
         const Link* x = &link;

         while (x->getRight())
         {
           s.insert(x);
           x = x->getRight();

           if (s.find(x) != s.end())
            return NONE;

           r += 1;
         }
      }

      {
         const Link* x = &link;

         while (x->getLeft())
         {
           s.insert(x);
           x = x->getLeft();

           if (s.find(x) != s.end())
            return NONE;

           l += 1;
         }
      }

      if (r > l)
         return RIGHT;
      else if (l > r)
         return LEFT;
      else
         return NONE;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   Link* left = new Link();
   Link* middle = new Link();
   Link* right = new Link();
   left->append(middle);
   middle->append(right);

   std::cout << Chain::whichSideIsLonger(*left);
}
#endif