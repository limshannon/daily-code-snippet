/*
   Question: TestDome - MultipleChoiceTest
   Solution by Arash Partow 2014

   Multiple choice test has several multiple choice questions. Each
   question can have only one correct answer. Additionally, timed
   multiple choice test can specify the time allowed for solving each
   question in the test.

   The code below satisfies this specification, but the customer
   complained that the memory usage of the program constantly increases.
   Fix this problem.
*/

#include <iostream>
#include <string>
#include <vector>

class MultipleChoiceTest
{
public:

   MultipleChoiceTest(int qc)
   : questionsCount(qc),
     answers(questionsCount,-1)
   {}

   virtual ~MultipleChoiceTest()
   {}

   void setAnswer(int questionIndex, int answer)
   {
      answers[questionIndex] = answer;
   }

   int getAnswer(int questionIndex) const
   {
      return answers[questionIndex];
   }

protected:

    int questionsCount;

private:

    std::vector<int> answers;
};


class TimedMultipleChoiceTest : public MultipleChoiceTest
{
public:

   TimedMultipleChoiceTest(int questionsCount)
   : MultipleChoiceTest(questionsCount),
     times(questionsCount,0)
   {}

   void setTime(int questionIndex, int time)
   {
      times[questionIndex] = time;
   }

   int getTime(int questionIndex) const
   {
      return times[questionIndex];
   }

private:

    std::vector<int> times;
};


#ifndef RunTests
void executeTest()
{
    MultipleChoiceTest test(5);
    for (int i = 0; i < 5; i++)
    {
        test.setAnswer(i, i);
    }

    for (int i = 0; i < 5; i++)
    {
        std::cout << "Question " << i + 1 << ", correct answer: " << test.getAnswer(i) << "\n";
    }
}

int main()
{
    for (int i = 0; i < 3; i++)
    {
        std::cout << "Test: " << i + 1 << "\n";
        executeTest();
    }
}
#endif
