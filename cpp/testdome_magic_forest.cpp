/*
   Question: TestDome - MagicForest
   Solution by Arash Partow 2014

   A magic forest is a set of nodes and edges. An edge connects two
   distinct nodes. Two nodes cannot be connected by more than one edge. A
   subset of nodes is a tree if it has the following two properties:

   1. For any two nodes in the subset there is exactly one series of
      edges Xi connecting them.

   2. There is no edge connecting a node from the subset to a node outside
      the subset.

   Write a class that can calculate the number of trees in a magic forest.

   For example, if we have a magic forest with 10 nodes (0-9) and edges
   1-2, 3-4, 3-5, 4-5, 6-7, 6-8, and 6-9, countTrees should return 3 as
   there are three trees (0), (1, 2), and (6, 7, 8, 9) in that magic
   forest. A subset of nodes (3, 4, 5) is not a tree since there are two
   series of edges connecting each two nodes. Nodes 3 and 5 are connected
   via direct edge 3-5 and via series of edges 3-4 and 4-5.
*/

#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Edge
{
public:

   Edge(int from, int to)
   {
      this->from = from;
      this->to = to;
   }

   int GetFrom() const
   {
      return from;
   }

   int GetTo() const
   {
      return to;
   }

private:

   int from;
   int to;
};

class MagicForest
{
private:

   typedef std::unordered_set<int> st;
   typedef std::shared_ptr<std::pair<st,bool>> st_t;

   std::unordered_map<int,st_t> st_map;

   bool si(const st& x0, const st& x1)
   {
      for (auto& i : x0)
      {
         if (x1.find(i) != x1.end())
            return true;
      }

      return false;
   }

   int tree_count;

public:
   /**
   *   \brief Initializes a new instance of the MagicForest class.
   *
   *   \param nodes Number of nodes in the magic forest. Nodes are numbered 0 .. nodes-1.
   *   \param edges List of edges.
   *
   **/
   MagicForest(int nodes, const std::vector<Edge>& edges)
   {
      for (auto e : edges)
      {
         int x = e.GetFrom();
         int y = e.GetTo();

         auto x_it = st_map.find(x);
         auto y_it = st_map.find(y);

         if (x_it == st_map.end() && y_it == st_map.end())
         {
            auto s = std::make_shared<std::pair<st,bool>>();

            s->second = true;

            s->first.insert(x);
            s->first.insert(y);

            st_map[x] = s;
            st_map[y] = s;
         }
         else if (x_it != st_map.end() && y_it == st_map.end())
         {
            auto s = x_it->second;

            s->first.insert(y);

            st_map[y] = s;
         }
         else if (x_it == st_map.end() && y_it != st_map.end())
         {
            auto s = y_it->second;

            s->first.insert(x);

            st_map[x] = s;
         }
         else
         {
            auto sx = x_it->second;
            auto sy = y_it->second;

            if (si(sx->first,sy->first))
            {
               sx->second = false;
            }

            std::copy(sy->first.begin(),sy->first.end(),std::inserter(sx->first,sx->first.end()));

            for (auto i : sy->first)
            {
               st_map[i] = sx;
            }
         }
      }

      std::unordered_set<st_t> t;

      for(auto& i : st_map)
      {
         t.insert(i.second);
      }

      tree_count = 0;
      int sz     = 0;

      for(auto& i : t)
      {
         sz += i->first.size();

         if (i->second)
            tree_count += 1;
      }

      if (sz < nodes)
         tree_count += (nodes - sz);
   }

   int countTrees() const
   {
      return tree_count;
   }
};


#ifndef RunTests
int main(int argc, const char* argv[])
{
   std::vector<Edge> edges;
   edges.push_back(Edge(1, 2));
   edges.push_back(Edge(3, 4));
   edges.push_back(Edge(3, 5));
   edges.push_back(Edge(4, 5));
   edges.push_back(Edge(6, 7));
   edges.push_back(Edge(6, 8));
   edges.push_back(Edge(6, 9));

   MagicForest forest(10, edges);

   std::cout << forest.countTrees();
}
#endif
