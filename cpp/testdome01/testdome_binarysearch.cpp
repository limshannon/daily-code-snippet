/*
   Question: TestDome - BinarySearchTree
   Solution by Arash Partow 2014

   Binary search tree (BST) is a binary tree where the value of each node
   is larger or equal to the values in all the nodes in that node's left
   subtree and is smaller than the values in all the nodes in that node's
   right subtree.

   Write a function that checks if a given binary search tree contains a
   given value.

   For example, for the following tree:

    n1 (Value: 1, Left: null, Right: null)
    n2 (Value: 2, Left: n1, Right: n3)
    n3 (Value: 3, Left: null, Right: null)

   Call to contains(n2, 3) should return true since a tree with root at
   n2 contains number 3.
*/

#include <stdexcept>
#include <string>
#include <iostream>

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

   static bool contains(const Node& root, int value)
   {
      if (root.getValue() == value)
         return true;
      else if (root.getLeft() && (value < root.getValue()))
         return contains(*root.getLeft(),value);
      else if (root.getRight() && (value > root.getValue()))
         return contains(*root.getRight(),value);
      else
         return false;
   }
};

#ifndef RunTests
int main()
{
    Node n1(1, NULL, NULL);
    Node n3(3, NULL, NULL);
    Node n2(2, &n1, &n3);

    std::cout << BinarySearchTree::contains(n2, 3);
}
#endif
