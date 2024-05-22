/*
   Question: TestDome - DataStruct
   Solution by Arash Partow 2014

   Design the data structure for a GPS navigation software that needs to
   efficiently handle two operations used with similar frequency: adding
   or updating the distance of a newly discovered shortest route to a
   city and polling (retrieving and removing) the city with the shortest
   distance.

   For example, if the first route found is to city 1 with distance 3,
   the second route found is to city 2 with distance 2 and the third
   route found is to city 1 with distance 1 then the first call to
   pollClosestCity should return 1 as city 1 is the closest city, while
   the second call to pollClosestCity should return 2 since city 2 is the
   closest after city 1 has been removed.
*/

#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>

class DataStruct
{
private:

   struct c_t
   {
      int id,d;
      bool v;
   };

   typedef std::vector<c_t*> q_t;

   std::unordered_map<int,c_t> m;
   q_t q;

   void insert(q_t& v, c_t* value)
   {
      if (v.size() == 0)
      {
         d_l = value->d;
         d_s = value->d;

         v.push_back(value);

         return;
      }
      else if (value->d <= d_s)
      {
         d_s = value->d;

         v.push_back(value);

         return;
      }
      else if (value->d >= d_l)
      {
         d_l = value->d;

         v.insert(v.begin(),value);

         return;
      }

      q_t::iterator it =
         std::lower_bound(v.begin(),v.end(),
           value,[](const c_t* l, const c_t* r) {return l->d > r->d;});
      v.insert(it,value);
   }

   int d_l;
   int d_s;

public:

   /**
   *   \brief Adds new city with new distance or updates existing city with shorter distance.
   *
   *   \param cityId City identifier.
   *   \param distance Distance.
   *
   **/
   void addOrUpdate(int cityId, int distance)
   {
      auto i = m.find(cityId);

      if (i == m.end())
      {
         c_t c;
         c.id = cityId;
         c.d  = distance;
         c.v  = true;

         m[cityId] = c;

         insert(q,&(m[cityId]));
      }
      else
      {
         c_t& c = i->second;

         int od = c.d;

         if (!c.v)
         {
            c.d = distance;
            c.v = true;

            insert(q,&c);

            return;
         }

         if (distance == od)
            return;

         auto lower = std::lower_bound(q.begin(),q.end(),c.d,[](const c_t* c,int d) {return d < c->d; });

         std::size_t i = std::distance(q.begin(), lower);

         for ( ; i < q.size(); ++i)
         {
            if (q[i]->id == c.id)
            {
               break;
            }
         }

         if (i == 0 && d_l < distance)
            d_l = distance;
         else if (i == (q.size() - 1) && d_s > distance)
            d_s = distance;
         else if (distance < d_s)
         {
            rotate_left(q.begin() + i, q.end(),1);

            q.back() = &c;

            d_s = distance;
         }
         else if (distance > d_l)
         {
            rotate_right(q.begin(), q.begin() + i + 1,1);

            q.front() = &c;

            d_l = distance;
         }
         else if (distance < od)
         {
            auto nl = std::upper_bound(q.begin() + i ,q.end(),distance,[](int d, const c_t* c) {return c->d < d; });
            std::size_t ni = std::distance(q.begin(), nl);

            std::rotate(q.begin() + i, q.begin() + i + 1, q.begin() + ni);
         }
         else
         {
            auto nl = std::upper_bound(q.begin(), q.begin() + i + 1,distance,[](int d, const c_t* c) {return c->d < d; });
            std::size_t ni = std::distance(q.begin(), nl);

            std::rotate(q.begin() + ni,q.begin() +  i,q.begin() +  i + 1);
         }

         c.d = distance;

         return;
      }
   }

   /**
   *   \brief Retrieves and removes city with smallest distance and returns it's id.
   *
   *   \return The closest city identifier.
   *
   **/
   int pollClosestCity()
   {
      c_t* c = q.back();
      c->v   = false;

      int cid = c->id;

      q.pop_back();

      if (q.size() > 0)
         d_s = q.back()->d;

      return cid;
   }

   template <typename Iterator>
   void rotate_left(Iterator begin, Iterator end, std::size_t n)
   {
      auto shift = n % distance(begin,end);
      std::rotate(begin, begin + shift, end);
   }

   template <typename Iterator>
   void rotate_right(Iterator begin, Iterator end, std::size_t n)
   {
      auto dist  = distance(begin,end);
      auto shift = dist - (n % dist);
      rotate_left(begin, end, shift);
   }
};

#ifndef RunTests
int main(int argc, const char* argv[])
{
   DataStruct ds;

   ds.addOrUpdate(1, 3);
   ds.addOrUpdate(2, 2);
   ds.addOrUpdate(1, 1);

   std::cout << ds.pollClosestCity();
   std::cout << '\n';
   std::cout << ds.pollClosestCity();

   return 0;
}
#endif
