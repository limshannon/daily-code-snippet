/*
   Question: TestDome - BinarySearchTree
   Solution by Arash Partow 2014

   Write a function that checks if a given binary tree is a valid
   binary search tree. A binary search tree (BST) is a binary tree
   where the value of each node is larger or equal to the values in
   all the nodes in that node's left subtree and is smaller than the
   values in all the nodes in that node's right subtree.

   For example, for the following tree:

   n1 (Value: 1, Left: null, Right: null)
   n2 (Value: 2, Left: n1, Right: n3)
   n3 (Value: 3, Left: null, Right: null)

   Call to isValidBST(n2) should return true since a tree with root at
   n2 is a valid binary search tree.

   Explanation: Subtrees rooted at nodes n1 and n3 are valid binary
   search trees as they have no children. A tree rooted at node n2 is
   a valid binary search tree since its value (2) is larger or equal
   to the largest value in its left subtree (1, rooted at n1) and is
   smaller than the smallest value in its right subtree (3 rooted at
   n3).
*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

class Node
{
public:
   Node(int value, Node* left, Node* right)
   {
      this->value = value;
      this->left = left;
      this->right = right;
   }

   int getValue() const
   {
      return value;
   }

   Node* getLeft() const
   {
      return left;
   }

   Node* getRight() const
   {
      return right;
   }

private:
   int value;
   Node* left;
   Node* right;
};

class BinarySearchTree
{
public:

   static bool isValidBST(const Node& root)
   {
      if (!isValidBST_left_impl(root.getLeft(),root.getValue()))
         return false;
      else if (!isValidBST_right_impl(root.getRight(),root.getValue()))
         return false;
      else if (root.getLeft () && (root.getLeft()->getValue() >= root.getValue()))
         return false;
      else if (root.getRight() && (root.getRight()->getValue() < root.getValue()))
         return false;
      return true;
   }

private:

   static bool isValidBST_right_impl(const Node* root, int parent_v)
   {
      if (root == 0)
         return true;

      if (!branch_leftright(root))
         return false;

      if (root->getLeft() && root->getRight())
         return parent_v < std::min(root->getLeft()->getValue(), root->getRight()->getValue());
      else if (root->getLeft())
         return parent_v < root->getLeft()->getValue();
      else if (root->getRight())
         return parent_v < root->getRight()->getValue();

      return parent_v < root->getValue();
   }

   static bool isValidBST_left_impl(const Node* root, int parent_v)
   {
      if (root == 0)
         return true;

      if (!branch_leftright(root))
         return false;

      if (root->getLeft() && root->getRight())
         return parent_v >= std::max(root->getLeft()->getValue(), root->getRight()->getValue());
      else if (root->getLeft())
         return parent_v >= root->getLeft()->getValue();
      else if (root->getRight())
         return parent_v >= root->getRight()->getValue();

      return parent_v >= root->getValue();
   }

   static bool branch_leftright(const Node* root)
   {
      if (!isValidBST_left_impl(root->getLeft(),root->getValue()))
         return false;
      else if (!isValidBST_right_impl(root->getRight(),root->getValue()))
         return false;

      return true;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   Node n1(1, NULL, NULL);
   Node n3(3, NULL, NULL);
   Node n2(2, &n1, &n3);

   std::cout << BinarySearchTree::isValidBST(n2);

   return 0;
}
#endif
