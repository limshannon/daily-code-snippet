#include <stdexcept>
#include <iostream>
#include <vector>
#include <utility>

std::pair<int, int> findTwoSum(const std::vector<int>& list, int sum)
{
    std::unordered_map<int, int> sumMap;
    //for (int i =0; i<list.size(); ++i){
    for (std::size_t i =0; i<list.size(); ++i){
        auto finding = sumMap.find(sum - list[i]);
        if (finding != sumMap.end())
            return std::make_pair(i, finding->second);
        sumMap[list[i]] = i;
        
    }
    return std::make_pair(-1,-1);
}

#ifndef RunTests
int main()
{
    std::vector<int> list = {3, 1, 5, 7, 5, 9};
    std::pair<int, int> indices = findTwoSum(list, 10);
    std::cout << indices.first << '\n' << indices.second;
}
#endif
