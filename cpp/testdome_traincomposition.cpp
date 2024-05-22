/*
   Question: TestDome - TrainComposition
   Solution by Arash Partow 2014

   A TrainComposition is built by attaching and detaching wagons from the
   left and the right sides.

   For example, if we start by attaching wagon 7 from the left followed
   by attaching wagon 13, again from the left, we get a composition of
   two wagons (13 and 7 from left to right). Now the first wagon that can
   be detached from the right is 7 and the first that can be detached
   from the left is 13.

   Implement a TrainComposition that models this problem.
*/

#include <stdexcept>
#include <iostream>
#include <deque>

class TrainComposition
{
public:

   void attachWagonFromLeft(int wagonId)
   {
      train_.push_front(wagonId);
   }

   void attachWagonFromRight(int wagonId)
   {
      train_.push_back(wagonId);
   }

   int detachWagonFromLeft()
   {
      int result = train_.front();
      train_.pop_front();
      return result;
   }

   int detachWagonFromRight()
   {
      int result = train_.back();
      train_.pop_back();
      return result;
   }

private:

    std::deque<int> train_;
};

#ifndef RunTests
int main()
{
    TrainComposition tree;
    tree.attachWagonFromLeft(7);
    tree.attachWagonFromLeft(13);
    std::cout << tree.detachWagonFromRight() << "\n"; // 7
    std::cout << tree.detachWagonFromLeft(); // 13
}
#endif
