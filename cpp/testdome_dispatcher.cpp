/*
   Question: TestDome - Dispatcher
   Solution by Arash Partow 2014

   A customer complained that the memory usage of the following component
   constantly increases although the number of workers and tasks remains
   constant.

   Fix the problem.
*/

#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>

class Worker
{
public:

   Worker(int id)
   {
      this->id = id;
   }

   void performTask(std::string task)
   {
      tasks.push_back(task);
   }

   size_t countTasks() const
   {
      return tasks.size();
   }

   std::string getTask(int index) const
   {
      return tasks.at(index);
   }

private:

   std::vector<std::string> tasks;
   int id;
};

class Dispatcher
{
public:

   Worker* acquireWorker(int id)
   {
      std::unordered_map<int, Worker*>::iterator search = workers.find(id);

      if (search == workers.end())
      {
         Worker* worker = new Worker(id);
         std::pair<int, Worker*> keyValue(id, worker);

         workers.insert(keyValue);

         return worker;
      }
      else
      {
         return search->second;
      }
   }

   void releaseWorker(int id)
   {
      delete workers[id];
      workers.erase(id);
   }

private:

   std::unordered_map<int, Worker*> workers;
};


#ifndef RunTests
void printWorker(const Worker& worker)
{
   for (size_t i = 0; i < worker.countTasks(); i++)
   {
      if (i != 0)
      {
         std::cout << ", ";
      }

      std::cout << worker.getTask(i);
   }
   std::cout << '\n';
}

int main(int argc, const char* argv[])
{
   Dispatcher d;

   d.acquireWorker(1)->performTask("Task11");
   d.acquireWorker(2)->performTask("Task21");
   printWorker(*d.acquireWorker(2));
   d.releaseWorker(2);

   d.acquireWorker(1)->performTask("Task12");
   printWorker(*d.acquireWorker(1));
   d.releaseWorker(1);

   return 0;
}
#endif
