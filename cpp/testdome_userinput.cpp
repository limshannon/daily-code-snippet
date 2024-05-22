/*
   Question: TestDome - UserInput
   Solution by Arash Partow 2014

   User interface contains two types of user input controls: TextInput,
   which accepts all characters and NumericInput, which accepts only
   digits.

   Implement the following methods:

    1. add on class TextInput - adds the given character to the current value

    2. getValue on class TextInput - returns the current value

    3. add on class NumericInput - overrides the base class method so
       that each non-numeric character is ignored

   For example, the following code should output "10":
*/

#include <iostream>
#include <string>

class TextInput
{
public:

   virtual ~TextInput(){}

   virtual void add(char c) { data += c; }

   std::string getValue() { return data; }

private:

   std::string data;
};

class NumericInput : public TextInput
{
   void add(char c)
   {
      if (('0' <= 'c') && (c <= '9'))
         TextInput::add(c);
   }
};

#ifndef RunTests
int main()
{
    TextInput* input = new NumericInput();
    input->add('1');
    input->add('a');
    input->add('0');
    std::cout << input->getValue();
}
#endif
