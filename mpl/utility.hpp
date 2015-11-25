#if !(defined MPL_UTILITY_HPP)

#define MPL_UTILITY_HPP

#include <iterator>

namespace mpl {
  
  namespace detail {
    
    template<typename T>
    struct iterator_traits : public std::iterator_traits<T> {
      typedef typename std::iterator_traits<T>::value_type insert_type;
    };
    
    template<typename T>
    struct iterator_traits<std::back_insert_iterator<T>> : public std::iterator_traits<std::back_insert_iterator<T>> {
      typedef typename T::value_type insert_type;
    };
    
  }
  
}

#endif
