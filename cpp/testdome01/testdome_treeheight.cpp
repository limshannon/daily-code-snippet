/*
   Question: TestDome - TreeHeight
   Solution by Arash Partow 2014

   A tree is an abstract data structure consisting of nodes. Each node
   has only one parent node and zero or more child nodes. Each tree has
   one special node, called a root, which has no parent node. A node is
   called a leaf if it has no child nodes. The height of a tree is the
   largest distance (number of parent-child connections) from the root to
   a leaf.

   Write a function that calculates the height of a given tree. For
   example, the tree below has a height of 2 since the longest distance
   (number of parent-child connections) from the root to a leaf is 2 (the
   connections are the root to a node and then the node to leaf 1):
*/

#include <stdexcept>
#include <iostream>
#include <algorithm>

class TreeNode
{
public:

   TreeNode(TreeNode* leftChild, TreeNode* rightChild)
   {
      this->leftChild = leftChild;
      this->rightChild = rightChild;
   }

   TreeNode* getLeftChild() const
   {
      return this->leftChild;
   }

   TreeNode* getRightChild() const
   {
      return this->rightChild;
   }

private:

   TreeNode* leftChild;
   TreeNode* rightChild;
};

class TreeHeight
{
public:

   static int calculateHeight(const TreeNode& root)
   {
      return calculateHeightx(root);
   }

private:

   static int calculateHeightx(const TreeNode& root)
   {
      int hl = 0;
      int hr = 0;

      if (root.getLeftChild() != NULL)
         hl = calculateHeightx(*root.getLeftChild()) + 1;

      if (root.getRightChild() != NULL)
         hr = calculateHeightx(*root.getRightChild()) + 1;

      return std::max(hl,hr);
   }

};

#ifndef RunTests
int main(int argc, const char* argv[])
{
   TreeNode* leaf1 = new TreeNode(NULL, NULL);
   TreeNode* leaf2 = new TreeNode(NULL, NULL);
   TreeNode* node = new TreeNode(leaf1, NULL);
   TreeNode* root = new TreeNode(node, leaf2);

   std::cout << TreeHeight::calculateHeight(*root);
}
#endif
