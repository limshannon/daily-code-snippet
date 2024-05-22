#include <vector>
#include <stdexcept>
#include <iostream>

int countNumbers(const std::vector<int>& sortedVector, int lessThan)
{
    throw std::logic_error("Waiting to be implemented");
}

#ifndef RunTests
int main()
{
    std::vector<int> v { 1, 3, 5, 7 };
    std::cout << countNumbers(v, 4);
}
#endif
