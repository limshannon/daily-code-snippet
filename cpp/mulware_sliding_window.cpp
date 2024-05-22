#include <iostream>
#include <vector>

std::vector<int> simulate(const std::vector<int> &entries)
{
    if (entries.size() <4)
        return entries;
    std::vector<int> arr {entries}; #copy for result 
    std::vector<std::vector<int>::iterator> tempEntries{};
    for (auto itX = std::begin(arr); itX!=std::end(arr) -4; ++itX){
        const auto rightT = itX +4;
        if (*itX <= *rightT)
            tempEntries.push_back(itX);        
    }
    for (auto itX = std::begin(arr) + 3; itX != std::end(arr); ++itX) {
        const auto leftT = itX - 3;
        if (*itX <= *leftT)
            tempEntries.push_back(itX);
    }
    for (auto it = std::begin(arr); it != std::end(arr); ++it) {
        if (std::find(std::begin(tempEntries), std::end(tempEntries), it) != std::end(tempEntries))
            *it = 0;
    }
    return arr;       
    //throw std::logic_error("Waiting to be implemented");
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
