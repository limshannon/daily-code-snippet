/*
   Question: TestDome - Grasshopper
   Solution by Arash Partow 2014

   Imagine a row of leaves (numbered 1 to N) and a grasshopper feeding on
   leaf X. After he eats leaf X, he will move either two leaves to the
   left or two leaves to the right, skipping the leaves he ate. The
   grasshopper will always move to an existing leaf.

   Implement a class that models the grasshopper's behaviour and can
   report its location at any time.

   For example, Grasshopper(5, 2) creates a scenario with a row of 5
   leaves (numbered 1 to 5) and a grasshopper feeding on leaf 2 (1 2* 3 4
   5). A call to eatAndHopRight() marks the current leaf (leaf 2) as
   eaten and moves the grasshopper two leaves to the right to leaf 4 (1 3
   4* 5). A call to eatAndHopLeft() marks the current leaf (leaf 4) as
   eaten and moves the grasshopper two leaves to the left to leaf 1 (1* 3
   5), skipping leaves marked as eaten.
*/

#include <vector>
#include <stdexcept>
#include <iostream>


class Grasshopper
{
public:

   Grasshopper(int n, int position)
   : pos(position)
   {
      loc.resize(n + 2);

      for (std::size_t i = 0; i < loc.size(); ++i)
      {
         loc[i].first = false;
         loc[i].second.first  = i - 1;
         loc[i].second.second = i + 1;
      }
   }

   /**
   *   \brief Grasshopper has eaten the current leaf and hopped left.
   *
   **/
   void eatAndHopLeft()
   {
      mark();
      pos = loc[loc[pos].second.first].second.first;
   }

   /**
   *   \brief Grasshopper has eaten the current leaf and hopped right.
   *
   **/
   void eatAndHopRight()
   {
      mark();
      pos = loc[loc[pos].second.second].second.second;
   }

   /**
   *   \return Leaf number that grasshopper is feeding on right now.
   *
   **/
   int whereAmI()
   {
      return pos;
   }

private:

   void mark()
   {
      loc[pos].first = true;
      loc[loc[pos].second.second].second.first  = loc[pos].second.first;
      loc[loc[pos].second.first ].second.second = loc[pos].second.second;
   }

   std::vector<std::pair<bool,std::pair<int,int>>> loc;
   int pos;

};

#ifndef RunTests
int main(int argc, const char* argloc[])
{
   Grasshopper g (5, 2);
   std::cout << g.whereAmI() << '\n';

   g.eatAndHopRight();
   std::cout << g.whereAmI() << '\n';

   g.eatAndHopLeft();
   std::cout << g.whereAmI() << '\n';
}
#endif

