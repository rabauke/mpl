try {
  mpl::file file;
  file.open(comm_world, "file_name.bin",
            mpl::file::access_mode::create | mpl::file::access_mode::read_write);
  // further file operations
  file.close();
} catch (mpl::io_failure &error) {
  // catch and handle i/o failures
  std::cerr << error.what() << '\n';
}
