/*
Singleton template class
*/

#ifndef SINGLETON_H_
#define SINGLETON_H_

template<typename C>

class Singleton{

public:
  static C* getInstance(){
    if(!m_instance){
      m_instance = new C();
    }
    return m_instance;
  };

  virtual ~Singleton(){};

private:
  static C* m_instance;

protected:
  Singleton(){};
  Singleton(Singleton& dontcopy){};

};

template <typename C> C* Singleton <C>::m_instance = 0;

#endif