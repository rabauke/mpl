#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <mpl/mpl.hpp>


template<typename T>
bool read_at_write_at_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at(comm_world.rank(), val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at(comm_world.rank(), val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_at_write_at_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at(comm_world.rank() * layout.extent(), val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at(comm_world.rank() * layout.extent(), val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_at_iwrite_at_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_at(comm_world.rank(), val)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_at(comm_world.rank(), val2)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_at_iwrite_at_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_at(comm_world.rank() * layout.extent(), val.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_at(comm_world.rank() * layout.extent(), val2.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_write_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.write(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.read(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_write_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.write(val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.read(val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_iwrite_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iwrite(val)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iread(val2)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_iwrite_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iwrite(val.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iread(val2.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_shared_write_shared_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_shared(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_shared(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_shared_write_shared_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_shared(val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_shared(val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_shared_iwrite_shared_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_shared(val)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_shared(val2)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_shared_iwrite_shared_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_shared(val.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_shared(val2.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_at_all_write_at_all_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at_all(comm_world.rank(), val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at_all(comm_world.rank(), val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_at_all_write_at_all_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at_all(comm_world.rank() * layout.extent(), val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at_all(comm_world.rank() * layout.extent(), val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_at_all_iwrite_at_all_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_at_all(comm_world.rank(), val)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_at_all(comm_world.rank(), val2)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_at_all_iwrite_at_all_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    auto r{file.iwrite_at_all(comm_world.rank() * layout.extent(), val.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    auto r{file.iread_at_all(comm_world.rank() * layout.extent(), val2.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_all_write_all_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.write_all(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.read_all(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_all_write_all_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.write_all(val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.read_all(val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_all_iwrite_all_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iwrite_all(val)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iread_all(val2)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool iread_all_iwrite_all_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iwrite_all(val.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    auto r{file.iread_all(val2.data(), layout)};
    r.wait();
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_ordered_write_ordered_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_ordered(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_ordered(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_ordered_write_ordered_test(const std::vector<T> &val, const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_ordered(val.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_ordered(val2.data(), layout);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_at_all_split_write_at_all_split_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at_all_begin(comm_world.rank(), val);
    file.write_at_all_end(val);
    // file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at_all_begin(comm_world.rank(), val2);
    file.read_at_all_end(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_at_all_split_write_at_all_split_test(const std::vector<T> &val,
                                               const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_at_all_begin(comm_world.rank() * layout.extent(), val.data(), layout);
    file.write_at_all_end(val.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_at_all_begin(comm_world.rank() * layout.extent(), val2.data(), layout);
    file.read_at_all_end(val2.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_all_split_write_all_split_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.write_all_begin(val);
    file.write_all_end(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{1, comm_world.rank()}});
    file.set_view("native", l);
    file.read_all_begin(val2);
    file.read_all_end(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_all_split_write_all_split_test(const std::vector<T> &val,
                                         const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.write_all_begin(val.data(), layout);
    file.write_all_end(val.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    mpl::indexed_layout<T> l({{layout.extent(), layout.extent() * comm_world.rank()}});
    file.set_view("native", l);
    file.read_all_begin(val2.data(), layout);
    file.read_all_end(val2.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_ordered_split_write_ordered_split_test(const T &val) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_ordered_begin(val);
    file.write_ordered_end(val);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  T val2;
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_ordered_begin(val2);
    file.read_ordered_end(val2);
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


template<typename T>
bool read_ordered_split_write_ordered_split_test(const std::vector<T> &val,
                                                 const mpl::layout<T> &layout) {
  auto filename{"test.bin"};
  const mpl::communicator &comm_world{mpl::environment::comm_world()};
  if (comm_world.rank() == 0)
    std::remove(filename);

  try {
    mpl::file file;
    file.open(comm_world, filename,
              mpl::file::access_mode::create | mpl::file::access_mode::read_write);
    file.set_view<T>("native");
    file.write_ordered_begin(val.data(), layout);
    file.write_ordered_end(val.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  std::vector<T> val2(val.size());
  try {
    mpl::file file;
    file.open(comm_world, filename, mpl::file::access_mode::read_only);
    file.set_view<T>("native");
    file.read_ordered_begin(val2.data(), layout);
    file.read_ordered_end(val2.data());
    file.close();
  } catch (mpl::error &error) {
    std::cerr << error.what() << '\n';
    return false;
  }

  return val2 == val;
}


SCENARIO("parallel MPI file input and output operations", "[file]") {
  [[maybe_unused]] const mpl::communicator &comm_world{mpl::environment::comm_world()};

  GIVEN("a parallel file opened across comm_world") {
    WHEN("performing positioned read_at and write_at operations") {
      THEN("data matches after writing and reading back") {
        REQUIRE(read_at_write_at_test(1.0));
        REQUIRE(read_at_write_at_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_at_write_at_test(std::vector{1.0, 2.0, 3.0}, mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing non-blocking positioned iread_at and iwrite_at operations") {
      THEN("data matches after completing asynchronous transfers") {
        REQUIRE(iread_at_iwrite_at_test(1.0));
        REQUIRE(iread_at_iwrite_at_test(std::array{1, 2, 3, 4}));
        REQUIRE(iread_at_iwrite_at_test(std::vector{1.0, 2.0, 3.0}, mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing individual file pointer read and write operations") {
      THEN("data matches after individual transfers") {
        REQUIRE(read_write_test(1.0));
        REQUIRE(read_write_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_write_test(std::vector{1.0, 2.0, 3.0}, mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing non-blocking individual file pointer iread and iwrite operations") {
      THEN("data matches after completing asynchronous individual transfers") {
        REQUIRE(iread_iwrite_test(1.0));
        REQUIRE(iread_iwrite_test(std::array{1, 2, 3, 4}));
        REQUIRE(iread_iwrite_test(std::vector{1.0, 2.0, 3.0}, mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing shared file pointer read_shared and write_shared operations") {
      THEN("data matches after shared pointer transfers") {
        REQUIRE(read_shared_write_shared_test(1.0));
        REQUIRE(read_shared_write_shared_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_shared_write_shared_test(std::vector{1.0, 2.0, 3.0},
                                              mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing non-blocking shared pointer iread_shared and iwrite_shared operations") {
      THEN("data matches after non-blocking shared pointer transfers") {
        REQUIRE(iread_shared_iwrite_shared_test(1.0));
        REQUIRE(iread_shared_iwrite_shared_test(std::array{1, 2, 3, 4}));
        REQUIRE(iread_shared_iwrite_shared_test(std::vector{1.0, 2.0, 3.0},
                                                mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing collective positioned read_at_all and write_at_all operations") {
      THEN("data matches after collective positioned transfers") {
        REQUIRE(read_at_all_write_at_all_test(1.0));
        REQUIRE(read_at_all_write_at_all_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_at_all_write_at_all_test(std::vector{1.0, 2.0, 3.0},
                                              mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing non-blocking collective positioned iread_at_all and iwrite_at_all operations") {
      THEN("data matches after non-blocking collective positioned transfers") {
        REQUIRE(iread_at_all_iwrite_at_all_test(1.0));
        REQUIRE(iread_at_all_iwrite_at_all_test(std::array{1, 2, 3, 4}));
        REQUIRE(iread_at_all_iwrite_at_all_test(std::vector{1.0, 2.0, 3.0},
                                                mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing collective read_all and write_all operations") {
      THEN("data matches after collective transfers") {
        REQUIRE(read_all_write_all_test(1.0));
        REQUIRE(read_all_write_all_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_all_write_all_test(std::vector{1.0, 2.0, 3.0},
                                        mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing non-blocking collective iread_all and iwrite_all operations") {
      THEN("data matches after non-blocking collective transfers") {
        REQUIRE(iread_all_iwrite_all_test(1.0));
        REQUIRE(iread_all_iwrite_all_test(std::array{1, 2, 3, 4}));
        REQUIRE(iread_all_iwrite_all_test(std::vector{1.0, 2.0, 3.0},
                                          mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing ordered collective read_ordered and write_ordered operations") {
      THEN("data matches after ordered transfers") {
        REQUIRE(read_ordered_write_ordered_test(1.0));
        REQUIRE(read_ordered_write_ordered_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_ordered_write_ordered_test(std::vector{1.0, 2.0, 3.0},
                                                mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing split collective read_at_all_split and write_at_all_split operations") {
      THEN("data matches after split collective positioned transfers") {
        REQUIRE(read_at_all_split_write_at_all_split_test(1.0));
        REQUIRE(read_at_all_split_write_at_all_split_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_at_all_split_write_at_all_split_test(std::vector{1.0, 2.0, 3.0},
                                                          mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing split collective read_all_split and write_all_split operations") {
      THEN("data matches after split collective transfers") {
        REQUIRE(read_all_split_write_all_split_test(1.0));
        REQUIRE(read_all_split_write_all_split_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_all_split_write_all_split_test(std::vector{1.0, 2.0, 3.0},
                                                    mpl::vector_layout<double>(3)));
      }
    }

    WHEN("performing split collective ordered read_ordered_split and write_ordered_split operations") {
      THEN("data matches after split collective ordered transfers") {
        REQUIRE(read_ordered_split_write_ordered_split_test(1.0));
        REQUIRE(read_ordered_split_write_ordered_split_test(std::array{1, 2, 3, 4}));
        REQUIRE(read_ordered_split_write_ordered_split_test(std::vector{1.0, 2.0, 3.0},
                                                           mpl::vector_layout<double>(3)));
      }
    }
  }
}
