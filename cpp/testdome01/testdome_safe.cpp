/*
   Question: TestDome - Safe
   Solution by Arash Partow 2014

   The Safe class is used as a simulation for a bank safe. It needs to
   satisfy the following conditions:

   The getItems() property should return the content of the safe.

   The addItem(item) method should add a new item to the safe. If the
   safe is full, std::logic_error should be thrown.

   The getDescription() method should return the safe's description.
   If the safe is full, it should return "Safe is full"; otherwise, it
   should return a message in the format "Safe: numOfItems/capacity".
   For example, if the capacity of the safe is two and one item is
   added, it should return: "Safe: 1/2".

   Fix the bugs!
*/

#include <stdexcept>
#include <vector>
#include <string>

class Safe
{
public:

   Safe(int capacity)
   {
      this->capacity = capacity;
   }

   int getCapacity() const
   {
      return capacity;
   }

   const std::vector<std::string>& getItems() const
   {
      return items;
   }

   void addItem(std::string item)
   {
      if (static_cast<int>(items.size()) >= capacity)
         throw std::logic_error("Overfilled");

      items.push_back(item);
   }

   std::string getDescription()
   {
      if (static_cast<int>(items.size()) == capacity)
         return "Safe is full";
      else
         return "Safe: " + std::to_string(items.size()) + "/" + std::to_string(capacity);
   }

private:

   std::vector<std::string> items;
   int capacity;
};
