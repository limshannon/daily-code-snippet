#include <iostream>
#include <vector>
#include <unordered_set>

std::vector<std::string> unique_names(const std::vector<std::string>& names1, const std::vector<std::string>& names2)
{
    std::unordered_set<string> set01;
    for (int i=0; i<names1.size(); i++)
        {
            if(auto iter=set01.find(names1[i]; iter != set01.send())
                continue;
            else{
                set01.insert(names1[i]);
            }
        }
    for (int i=0; i<names2.size(); i++)
        {
            if(auto iter=set01.find(names2[i]; iter != set01.send())
                continue;
            else{
                set01.insert(names2[i]);
            }
        }
    return set01;
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
