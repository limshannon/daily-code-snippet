#include <iostream>
#include <vector>

std::vector<int> simulate(const std::vector<int> &entries)
{
    throw std::logic_error("Waiting to be implemented");
}

#ifndef RunTests
int main()
{
    std::vector<int> result = simulate({1, 2, 0, 5, 0, 2, 4, 3, 3, 3});
    for (int value : result)
    {
        std::cout << value << " ";
    }
    // Expected output
    // 1, 0, 0, 5, 0, 0, 0, 3, 3, 0
}
#endif
