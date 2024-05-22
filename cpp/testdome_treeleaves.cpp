/*
   Question: TestDome - TreeLeaves
   Solution by Arash Partow 2014

   A tree is an abstract data structure consisting of nodes. Each node
   has only one parent node and zero or more child nodes. Each tree has
   one special node, called a root, which has no parent node. A node is
   called a leaf if it has no child nodes.

   A tree can be represented by an array P where P[i] is the parent of
   node i. For the root node r, P[r] equals -1.

   Write a function that calculates the number of leaves for a given
   tree.

   For example, the tree represented by the array {1, 3, 1, -1, 3} has 5
   nodes, 0 to 4, of which 3 nodes are leaves (0, 2 and 4 since all the
   other nodes have at least one child node).
*/

#include <vector>
#include <stdexcept>
#include <unordered_set>

class TreeLeaves
{
public:

   static int calculateNumOfLeaves(const std::vector<int>& tree)
   {
      std::unordered_set<int> parent_set;

      for (auto v : tree)
      {
         if (v != -1)
          parent_set.insert(v);
      }

      return (tree.size() - parent_set.size());
   }
};

#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::vector<int> tree;
   tree.push_back(1);
   tree.push_back(3);
   tree.push_back(1);
   tree.push_back(-1);
   tree.push_back(3);

   std::cout << Tree::calculateNumOfLeaves(tree);
}
#endif
