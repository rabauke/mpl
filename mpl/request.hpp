#ifndef MPL_REQUEST_HPP

#define MPL_REQUEST_HPP

#include <mpi.h>
#include <utility>
#include <optional>
#include <vector>


namespace mpl {

  class irequest;

  class prequest;

  class irequest_pool;

  class rrequest_pool;

  /// Indicates kind of outcome of test for request completion.
  enum class test_result {
    completed,          ///< some request has been completed
    no_completed,       ///< no request has been completed
    no_active_requests  ///< there is no request waiting for completion
  };

  namespace impl {

    template<typename T>
    class base_request;

    template<typename T>
    class request_pool;

    class base_irequest {
      MPI_Request request_{MPI_REQUEST_NULL};

    public:
      explicit base_irequest(MPI_Request request) : request_{request} {
      }

      friend class base_request<base_irequest>;

      friend class request_pool<base_irequest>;
    };

    class base_prequest {
      MPI_Request request_{MPI_REQUEST_NULL};

    public:
      explicit base_prequest(MPI_Request request) : request_{request} {
      }

      friend class base_request<base_prequest>;

      friend class request_pool<base_prequest>;
    };

    //------------------------------------------------------------------

    template<typename T>
    class base_request {
    protected:
      MPI_Request request_;

    public:
      base_request() = delete;

      base_request(const base_request &) = delete;

      explicit base_request(const base_irequest &req) : request_{req.request_} {
      }
      explicit base_request(const base_prequest &req) : request_{req.request_} {
      }

      base_request(base_request &&other) noexcept : request_(other.request_) {
        other.request_ = MPI_REQUEST_NULL;
      }

      ~base_request() {
        if (request_ != MPI_REQUEST_NULL)
          MPI_Request_free(&request_);
      }

      void operator=(const base_request &) = delete;

      base_request &operator=(base_request &&other) noexcept {
        if (this != &other) {
          if (request_ != MPI_REQUEST_NULL)
            MPI_Request_free(&request_);
          request_ = other.request_;
          other.request_ = MPI_REQUEST_NULL;
        }
        return *this;
      }

      /// Cancels the request if it is pending.
      void cancel() {
        if (request_ != MPI_REQUEST_NULL)
          detail::check(MPI_Cancel(&request_));
      }

      /// Tests for the completion.
      /// \return the operation's status if completed successfully
      std::optional<status_t> test() {
        int result{true};
        status_t s;
        detail::check(MPI_Test(&request_, &result, static_cast<MPI_Status *>(&s)));
        if (result != 0)
          return s;
        return {};
      }

      /// Wait for a pending communication operation.
      /// \return operation's status after completion
      status_t wait() {
        status_t s;
        detail::check(MPI_Wait(&request_, static_cast<MPI_Status *>(&s)));
        return s;
      }

      /// Access information associated with a request without freeing the request.
      /// \return the operation's status if completed successfully
      std::optional<status_t> get_status() {
        int result{true};
        status_t s;
        detail::check(MPI_Request_get_status(request_, &result, static_cast<MPI_Status *>(&s)));
        if (result != 0)
          return s;
        return {};
      }
    };

    //------------------------------------------------------------------

    template<typename T>
    class request_pool {
    protected:
      std::vector<MPI_Request> requests_;
      std::vector<status_t> statuses_;

    public:
      /// Type used in all index-based operations.
      using size_type = std::vector<MPI_Request>::size_type;

      request_pool() = default;

      request_pool(const request_pool &) = delete;

      request_pool(request_pool &&other) noexcept
          : requests_(std::move(other.requests_)), statuses_{std::move(other.statuses_)} {
      }

      ~request_pool() {
        for (auto &request : requests_)
          if (request != MPI_REQUEST_NULL)
            MPI_Request_free(&request);
      }

      void operator=(const request_pool &) = delete;

      request_pool &operator=(request_pool &&other) noexcept {
        if (this != &other) {
          for (auto &request : requests_)
            if (request != MPI_REQUEST_NULL)
              MPI_Request_free(&request);
          requests_ = std::move(other.requests_);
          statuses_ = std::move(other.statuses_);
        }
        return *this;
      }

      /// Determine the size of request pool.
      /// \return number of requests currently in request pool
      [[nodiscard]] size_type size() const {
        return requests_.size();
      }

      /// Determine if request pool is empty.
      /// \return true if number of requests currently in request pool is non-zero
      [[nodiscard]] bool empty() const {
        return requests_.empty();
      }

      /// Get status of a request.
      /// Status validity depends on the most recent completion call. In particular, a
      /// testall that returns false does not provide usable completion statuses, and the
      /// MPI error field is defined only for an MPI_ERR_IN_STATUS result.
      /// \param i index of the request for which the status will be returned
      /// \return status of request
      [[nodiscard]] const status_t &get_status(size_type i) const {
        return statuses_[i];
      }

      /// Cancels a pending request in the pool.
      /// \param i index of the request for which shall be cancelled
      void cancel(size_type i) {
        detail::check(MPI_Cancel(&requests_[i]));
      }

      /// Cancels all requests in the pool, stopping at the first error.
      void cancelall() {
        for (size_type i = 0; i < requests_.size(); ++i)
          cancel(i);
      }

      /// Wait for completion of a single pending communication operation in the pool.
      /// \param i index of the request to wait for
      /// \return operation's status after completion
      status_t wait(size_type i) {
        detail::check(MPI_Wait(&requests_[i], static_cast<MPI_Status *>(&statuses_[i])));
        return statuses_[i];
      }

      /// Tests for completion of a single pending communication operation in the pool.
      /// \param i index of the request to test
      /// \return the operation's status if completed successfully
      std::optional<status_t> test(size_type i) {
        int result{true};
        detail::check(
            MPI_Test(&requests_[i], &result, static_cast<MPI_Status *>(&statuses_[i])));
        if (result != 0)
          return statuses_[i];
        return {};
      }

      /// Move a request into the request pool.
      /// \param request request to move into the pool
      void push(T &&request) {
        requests_.push_back(request.request_);
        request.request_ = MPI_REQUEST_NULL;
        statuses_.push_back(status_t());
      }

      /// Wait for completion of any pending communication operation.
      /// \return pair containing the outcome of the wait operation and an index to the
      /// completed request if there was any pending request
      std::pair<test_result, size_type> waitany() {
        int index;
        status_t s;
        detail::check(
            MPI_Waitany(size(), requests_.data(), &index, static_cast<MPI_Status *>(&s)));
        if (index != MPI_UNDEFINED) {
          statuses_[index] = s;
          return std::make_pair(test_result::completed, static_cast<size_type>(index));
        }
        return std::make_pair(test_result::no_active_requests, size());
      }

      /// Test for completion of any pending communication operation.
      /// \return pair containing the outcome of the test and an index to the completed
      /// request if there was any pending request
      std::pair<test_result, size_type> testany() {
        int index, flag;
        status_t s;
        detail::check(MPI_Testany(size(), requests_.data(), &index, &flag,
                                  static_cast<MPI_Status *>(&s)));
        if (flag != 0 and index != MPI_UNDEFINED) {
          statuses_[index] = s;
          return std::make_pair(test_result::completed, static_cast<size_type>(index));
        }
        if (flag != 0 and index == MPI_UNDEFINED)
          return std::make_pair(test_result::no_active_requests, size());
        return std::make_pair(test_result::no_completed, size());
      }

      void store_some_statuses(const int return_code, const int count,
                               const std::vector<int> &out_indices,
                               const std::vector<status_t> &out_statuses) {
        if ((return_code == MPI_SUCCESS or return_code == MPI_ERR_IN_STATUS) and
            count != MPI_UNDEFINED)
          for (int i{0}; i < count; ++i)
            statuses_[out_indices[i]] = out_statuses[i];
      }

      template<typename S>
      static void check_statuses(const int return_code, const S *statuses, const int n,
                                 const int *indices = nullptr) {
        if (return_code != MPI_ERR_IN_STATUS) {
          detail::check(return_code);
          return;
        }
        // Requests with status MPI_ERR_PENDING are still pending and therefore neither
        // failed nor completed.
        std::vector<int> failed;
        std::vector<int> completed;
        int first_code{MPI_SUCCESS};
        for (int i{0}; i < n; ++i) {
          const int e{static_cast<const MPI_Status &>(statuses[i]).MPI_ERROR};
          const int index{indices != nullptr ? indices[i] : i};
          if (e == MPI_SUCCESS) {
            completed.push_back(index);
          } else if (e != MPI_ERR_PENDING) {
            if (failed.empty())
              first_code = e;
            failed.push_back(index);
          }
        }
        if (failed.empty())
          detail::check(return_code);
        throw request_error{first_code, std::move(failed), std::move(completed)};
      }

      /// Waits for completion of all pending requests.
      void waitall() {
        check_statuses(
            MPI_Waitall(size(), requests_.data(), static_cast<MPI_Status *>(statuses_.data())),
            statuses_.data(), size());
      }

      /// Tests for completion of all pending requests.
      /// \return true if all pending requests have completed
      bool testall() {
        int flag;
        check_statuses(MPI_Testall(size(), requests_.data(), &flag,
                                   static_cast<MPI_Status *>(statuses_.data())),
                       statuses_.data(), size());
        return static_cast<bool>(flag);
      }

      /// Waits until one or more pending requests have finished.
      /// \return pair containing the outcome of the wait operation and a list of indices to
      /// the completed requests if there was any pending request
      std::pair<test_result, std::vector<size_type>> waitsome() {
        std::vector<int> out_indices(size());
        std::vector<status_t> out_statuses(size());
        int count{MPI_UNDEFINED};
        const int return_code{MPI_Waitsome(size(), requests_.data(), &count, out_indices.data(),
                                           static_cast<MPI_Status *>(out_statuses.data()))};
        // Store the statuses first, because the requests that completed are already
        // deallocated by MPI, even if another request failed.
        store_some_statuses(return_code, count, out_indices, out_statuses);
        check_statuses(return_code, out_statuses.data(), count == MPI_UNDEFINED ? 0 : count,
                       out_indices.data());
        if (count != MPI_UNDEFINED) {
          return std::make_pair(
              test_result::completed,
              std::vector<size_t>(out_indices.begin(), out_indices.begin() + count));
        }
        return std::make_pair(test_result::no_active_requests, std::vector<size_t>{});
      }

      /// Tests if one or more pending requests have finished.
      /// \return pair containing the outcome of the test and a list of indices to the completed
      /// requests if there was any pending request
      std::pair<test_result, std::vector<size_type>> testsome() {
        std::vector<int> out_indices(size());
        std::vector<status_t> out_statuses(size());
        int count{MPI_UNDEFINED};
        const int return_code{MPI_Testsome(size(), requests_.data(), &count, out_indices.data(),
                                           static_cast<MPI_Status *>(out_statuses.data()))};
        // Store the statuses first, because the requests that completed are already
        // deallocated by MPI, even if another request failed.
        store_some_statuses(return_code, count, out_indices, out_statuses);
        check_statuses(return_code, out_statuses.data(), count == MPI_UNDEFINED ? 0 : count,
                       out_indices.data());
        if (count != MPI_UNDEFINED) {
          return std::make_pair(
              count == 0 ? test_result::no_completed : test_result::completed,
              std::vector<size_t>(out_indices.begin(), out_indices.begin() + count));
        }
        return std::make_pair(test_result::no_active_requests, std::vector<size_t>{});
      }
    };

  }  // namespace impl

  //--------------------------------------------------------------------

  /// Represents a non-blocking communication request.
  class irequest : public impl::base_request<impl::base_irequest> {
    using base = impl::base_request<impl::base_irequest>;
    using base::request_;

  public:
#if (!defined MPL_DOXYGEN_SHOULD_SKIP_THIS)
    irequest(const impl::base_irequest &r) : base{r} {
    }
#endif

    /// Deleted copy constructor.
    irequest(const irequest &) = delete;

    /// Move constructor.
    /// \param other the request to move from
    irequest(irequest &&other) noexcept : base{std::move(other)} {
    }

    /// Deleted copy operator.
    void operator=(const irequest &) = delete;

    /// Move operator.
    /// \param other the request to move from
    /// \return reference to the moved-to request
    irequest &operator=(irequest &&other) noexcept {
      base::operator=(std::move(other));
      return *this;
    }

    friend class impl::request_pool<irequest>;
  };

  //--------------------------------------------------------------------

  /// Container for managing a list of non-blocking communication requests.
  class irequest_pool : public impl::request_pool<irequest> {
    using base = impl::request_pool<irequest>;

  public:
    /// Constructs an empty pool of   communication requests.
    irequest_pool() = default;

    /// Deleted copy constructor.
    irequest_pool(const irequest_pool &) = delete;

    /// Move constructor.
    /// \param other the request pool to move from
    irequest_pool(irequest_pool &&other) noexcept : base{std::move(other)} {
    }

    /// Deleted copy operator.
    void operator=(const irequest_pool &) = delete;

    /// Move operator.
    /// \param other the request pool to move from
    /// \return reference to the moved-to request pool
    irequest_pool &operator=(irequest_pool &&other) noexcept {
      base::operator=(std::move(other));
      return *this;
    }
  };

  //--------------------------------------------------------------------

  /// Represents a persistent communication request.
  class prequest : public impl::base_request<impl::base_prequest> {
    using base = impl::base_request<impl::base_prequest>;
    using base::request_;

  public:
#if (!defined MPL_DOXYGEN_SHOULD_SKIP_THIS)
    prequest(const impl::base_prequest &r) : base{r} {
    }
#endif

    /// Deleted copy constructor.
    prequest(const prequest &) = delete;

    /// Move constructor.
    /// \param other the request to move from
    prequest(prequest &&other) noexcept : base{std::move(other)} {
    }

    /// Deleted copy operator.
    void operator=(const prequest &) = delete;

    /// Move operator.
    /// \param other the request to move from
    /// \return reference to the moved-to request
    prequest &operator=(prequest &&other) noexcept {
      base::operator=(std::move(other));
      return *this;
    }

    /// Start communication operation.
    void start() {
      detail::check(MPI_Start(&request_));
    }

    friend class impl::request_pool<prequest>;
  };

  //--------------------------------------------------------------------

  /// Container for managing a list of persisting communication requests.
  class prequest_pool : public impl::request_pool<prequest> {
    using base = impl::request_pool<prequest>;
    using base::requests_;

    enum class request_state { inactive, active };
    std::vector<request_state> active_requests_;

  public:
    using size_type = typename base::size_type;

  private:
    [[nodiscard]] bool has_active_requests() const {
      for (const auto active : active_requests_)
        if (active == request_state::active)
          return true;
      return false;
    }

    void mark_inactive(const size_type i) {
      if (active_requests_[i] == request_state::active)
        active_requests_[i] = request_state::inactive;
    }

    void mark_reported_requests_inactive(const request_error &error) {
      const auto mark{[this](const std::vector<int> &indices) {
        for (const int index : indices)
          if (index >= 0 and static_cast<size_type>(index) < active_requests_.size())
            mark_inactive(static_cast<size_type>(index));
      }};
      mark(error.completed_requests());
      mark(error.failed_requests());
    }

    void mark_all_inactive() {
      for (size_type i{0}; i < active_requests_.size(); ++i)
        mark_inactive(i);
    }

  public:
    /// Constructs an empty pool of persistent communication requests.
    prequest_pool() = default;

    /// Deleted copy constructor.
    prequest_pool(const prequest_pool &) = delete;

    /// Move constructor.
    /// \param other the request pool to move from
    prequest_pool(prequest_pool &&other) noexcept
        : base{std::move(other)}, active_requests_{std::move(other.active_requests_)} {
    }

    /// Deleted copy constructor.
    void operator=(const prequest_pool &) = delete;

    /// Move operator.
    /// \param other the request pool to move from
    prequest_pool &operator=(prequest_pool &&other) noexcept {
      if (this != &other) {
        base::operator=(std::move(other));
        active_requests_ = std::move(other.active_requests_);
      }
      return *this;
    }

    /// Move a persistent request into the request pool.
    /// \param request request to move into the pool
    void push(prequest &&request) {
      active_requests_.reserve(size() + 1);
      base::push(std::move(request));
      active_requests_.push_back(request_state::inactive);
    }

    /// Start a single persistent request in the pool.
    /// \param i index of the persistent request to start
    void start(size_type i) {
      detail::check(MPI_Start(&requests_[i]));
      active_requests_[i] = request_state::active;
    }

    /// Start all persistent requests in the pool.
    void startall() {
      detail::check(MPI_Startall(size(), requests_.data()));
      for (size_type i{0}; i < size(); ++i)
        active_requests_[i] = request_state::active;
    }

    /// Wait for completion of a single persistent communication operation.
    /// \param i index of the request to wait for
    /// \return operation's status after completion
    status_t wait(const size_type i) {
      const auto result{base::wait(i)};
      mark_inactive(i);
      return result;
    }

    /// Test for completion of a single persistent communication operation.
    /// \param i index of the request to test
    /// \return the operation's status if completed successfully
    std::optional<status_t> test(const size_type i) {
      const auto result{base::test(i)};
      if (result)
        mark_inactive(i);
      return result;
    }

    /// Wait for completion of any active persistent communication operation.
    /// \return outcome and index of the completed request
    std::pair<test_result, size_type> waitany() {
      if (not has_active_requests())
        return std::make_pair(test_result::no_active_requests, size());
      const auto result{base::waitany()};
      if (result.first == test_result::completed)
        mark_inactive(result.second);
      return result;
    }

    /// Test for completion of any active persistent communication operation.
    /// \return outcome and index of the completed request
    std::pair<test_result, size_type> testany() {
      if (not has_active_requests())
        return std::make_pair(test_result::no_active_requests, size());
      const auto result{base::testany()};
      if (result.first == test_result::completed)
        mark_inactive(result.second);
      return result;
    }

    /// Wait for completion of all active persistent communication operations.
    void waitall() {
      try {
        base::waitall();
      } catch (const request_error &error) {
        mark_reported_requests_inactive(error);
        throw;
      }
      mark_all_inactive();
    }

    /// Test for completion of all active persistent communication operations.
    /// \return true if all active requests have completed
    bool testall() {
      bool result;
      try {
        result = base::testall();
      } catch (const request_error &error) {
        mark_reported_requests_inactive(error);
        throw;
      }
      if (result)
        mark_all_inactive();
      return result;
    }

    /// Wait until one or more active persistent requests have completed.
    /// \return outcome and indices of the completed requests
    std::pair<test_result, std::vector<size_type>> waitsome() {
      if (not has_active_requests())
        return std::make_pair(test_result::no_active_requests, std::vector<size_type>{});
      try {
        auto result = base::waitsome();
        if (result.first == test_result::completed)
          for (const auto index : result.second)
            mark_inactive(index);
        return result;
      } catch (const request_error &error) {
        mark_reported_requests_inactive(error);
        throw;
      }
    }

    /// Test whether one or more active persistent requests have completed.
    /// \return outcome and indices of the completed requests
    std::pair<test_result, std::vector<size_type>> testsome() {
      if (not has_active_requests())
        return std::make_pair(test_result::no_active_requests, std::vector<size_type>{});
      try {
        auto result = base::testsome();
        if (result.first == test_result::completed)
          for (const auto index : result.second)
            mark_inactive(index);
        return result;
      } catch (const request_error &error) {
        mark_reported_requests_inactive(error);
        throw;
      }
    }
  };

}  // namespace mpl

#endif
