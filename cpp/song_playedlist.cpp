#include <stdexcept>
#include <iostream>
#include <unordered_set>

class Song
{
public:
    Song(std::string name): name(name), nextSong(NULL) {}
    
    void next(Song* song)
    {
        this->nextSong = song;
    }

    bool isInRepeatingPlaylist()
    {
        std::unordered_set<std::string> playedSongs;
        Song *pSong = this;
        while (pSong != nullptr){
            if (playedSongs.find(pSong->name)==playedSongs.end())
                playedSongs.insert(pSong->name);
            else
                return true;
            pSong=pSong->nextSong;
            
            
        }
        return false;
        
    }

private:
    const std::string name;
    Song* nextSong;
};

#ifndef RunTests
int main()
{
    Song* first = new Song("Hello");
    Song* second = new Song("Eye of the tiger");
    
    first->next(second);
    second->next(first);

    std::cout << std::boolalpha << first->isInRepeatingPlaylist();
}
#endif
