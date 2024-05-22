#include <vector>
#include <stdexcept>
#include <iostream>

int countNumbers(const std::vector<int>& sortedVector, int lessThan)
{
    auto res = std::lower_bound(std::begin(sortedVector), std::end(sortedVector), lessThan);
    return std::distance(std::begin(sortedVector), res);
}
/*{ //Performance test when sortedVector contains lessThan: Time limit exceeded 
    int res =0;
    for(std::size_t i=0; i< sortedVector.size(); ++i)
        {
            if (sortedVector[i]<lessThan){
              ++res;
            }
        }
    return res;
}*/

#ifndef RunTests
int main()
{
    std::vector<int> v { 1, 3, 5, 7 };
    std::cout << countNumbers(v, 4);
}
#endif
