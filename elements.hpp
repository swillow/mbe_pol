#ifndef W_ELEMENTS_HPP
#define W_ELEMENTS_HPP

#include <string>
#include <vector>

namespace willow {

struct element {
  unsigned short Z;
  std::string symbol;
  double mass;
};

static std::vector<element> element_info
{   {1, "H", 1.007825}, // 1H
    {1, "D", 2.014102}, // 2H
    {2, "He",4.002603}, // 2 He
    {3, "Li",7.016004},// 7Li
    {4, "Be",9.012182}, //9 Be
    {5, "B",11.009306}, // 11B
    {6, "C", 12.000}, // 12 C
    {7, "N", 14.003074}, // 14 N
    {8, "O", 15.994915} // 16 O
};

}  // namespace willow

#endif
