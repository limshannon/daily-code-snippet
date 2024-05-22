#include <iostream>
#include <vector>
#include <algorithm>


std::vector<std::string> unique_names(const std::vector<std::string>& names1, const std::vector<std::string>& names2)
{
    std::vector<std::string> res;
    res.insert(res.end(), names1.begin(), names1.end());
    res.insert(res.end(), names2.begin(), names2.end());
    
    std::sort(res.begin(), res.end());
    auto last = std::unique(res.begin(), res.end());
    res.erase(last, res.end());
     
    return res;
}

#ifndef RunTests
int main()
{
    std::vector<std::string> names1 = {"Ava", "Emma", "Olivia"};
    std::vector<std::string> names2 = {"Olivia", "Sophia", "Emma"};
    
    std::vector<std::string> result = unique_names(names1, names2);
    for(auto element : result)
    {
        std::cout << element << ' '; // should print Ava Emma Olivia Sophia
    }
}
#endif
