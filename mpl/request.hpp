#if !(defined MPL_REQUEST_HPP)

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
          MPI_Cancel(&request_);
      }

      /// Tests for the completion.
      /// \return the operation's status if completed successfully
      std::optional<status_t> test() {
        int result{true};
        status_t s;
        MPI_Test(&request_, &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Wait for a pending communication operation.
      /// \return operation's status after completion
      status_t wait() {
        status_t s;
        MPI_Wait(&request_, static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Access information associated with a request without freeing the request.
      /// \return the operation's status if completed successfully
      std::optional<status_t> get_status() {
        int result{true};
        status_t s;
        MPI_Request_get_status(request_, &result, static_cast<MPI_Status *>(&s));
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

      /// Tests for the completion for a request in the pool.
      /// \param i index of the request for which shall be tested
      /// \return the operation's status if completed successfully
      std::optional<status_t> test(size_type i) {
        int result{true};
        status_t s;
        MPI_Test(&requests_[i], &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Wait for a pending request in the pool.
      /// \param i index of the request for which shall be waited
      /// \return operation's status after completion
      status_t wait(size_type i) {
        status_t s;
        MPI_Wait(&requests_[i], static_cast<MPI_Status *>(&s));
        return s;
      }

      /// Access information associated with a request in the pool without freeing the request.
      /// \param i index of the request for which the status will be returned
      /// \return the operation's status if completed successfully
      std::optional<status_t> try_get_status(size_type i) {
        int result{true};
        status_t s;
        MPI_Request_get_status(requests_[i], &result, static_cast<MPI_Status *>(&s));
        if (result != 0)
          return s;
        return {};
      }

      /// Get status of a request.
      /// \param i index of the request for which the status will be returned
      /// \return status of request
      [[nodiscard]] const status_t &get_status(size_type i) const {
        return statuses_[i];
      }

      /// Cancels a pending request in the pool.
      /// \param i index of the request for which shall be cancelled
      void cancel(size_type i) {
        if (requests_[i] != MPI_REQUEST_NULL)
          MPI_Cancel(&requests_[i]);
      }

      /// Cancels all requests in the pool.
      void cancelall() {
        for (size_type i = 0; i < requests_.size(); ++i)
          cancel(i);
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
        MPI_Waitany(size(), &requests_[0], &index, static_cast<MPI_Status *>(&s));
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
        MPI_Testany(size(), &requests_[0], &index, &flag, static_cast<MPI_Status *>(&s));
        if (flag != 0 and index != MPI_UNDEFINED) {
          statuses_[index] = s;
          return std::make_pair(test_result::completed, static_cast<size_type>(index));
        }
        if (flag != 0 and index == MPI_UNDEFINED)
          return std::make_pair(test_result::no_active_requests, size());
        return std::make_pair(test_result::no_completed, size());
      }

      /// Waits for completion of all pending requests.
      void waitall() {
        MPI_Waitall(size(), &requests_[0], static_cast<MPI_Status *>(&statuses_[0]));
      }

      /// Tests for completion of all pending requests.
      /// \return true if all pending requests have completed
      bool testall() {
        int flag;
        MPI_Testall(size(), &requests_[0], &flag, static_cast<MPI_Status *>(&statuses_[0]));
        return static_cast<bool>(flag);
      }

      /// Waits until one or more pending requests have finished.
      /// \return pair containing the outcome of the wait operation and a list of indices to
      /// the completed requests if there was any pending request
      std::pair<test_result, std::vector<size_type>> waitsome() {
        std::vector<int> out_indices(size());
        std::vector<status_t> out_statuses(size());
        int count;
        MPI_Waitsome(size(), &requests_[0], &count, out_indices.data(),
                     static_cast<MPI_Status *>(&out_statuses[0]));
        if (count != MPI_UNDEFINED) {
          for (int i{0}; i < count; ++i)
            statuses_[out_indices[i]] = out_statuses[i];
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
        int count;
        MPI_Testsome(size(), &requests_[0], &count, out_indices.data(),
                     static_cast<MPI_Status *>(&out_statuses[0]));
        if (count != MPI_UNDEFINED) {
          for (int i{0}; i < count; ++i)
            statuses_[out_indices[i]] = out_statuses[i];
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
      MPI_Start(&request_);
    }

    friend class impl::request_pool<prequest>;
  };

  //--------------------------------------------------------------------

  /// Container for managing a list of persisting communication requests.
  class prequest_pool : public impl::request_pool<prequest> {
    using base = impl::request_pool<prequest>;
    using base::requests_;

  public:
    /// Constructs an empty pool of persistent communication requests.
    prequest_pool() = default;

    /// Deleted copy constructor.
    prequest_pool(const prequest_pool &) = delete;

    /// Move constructor.
    /// \param other the request pool to move from
    prequest_pool(prequest_pool &&other) noexcept : base{std::move(other)} {
    }

    /// Deleted copy constructor.
    void operator=(const prequest_pool &) = delete;

    /// Move operator.
    /// \param other the request pool to move from
    prequest_pool &operator=(prequest_pool &&other) noexcept {
      base::operator=(std::move(other));
      return *this;
    }

    /// Start a persistent requests in the pool.
    /// \param i index of the request for which shall be started
    void start(size_type i) {
      MPI_Start(&requests_[i]);
    }

    /// Start all persistent requests in the pool.
    void startall() {
      MPI_Startall(size(), &requests_[0]);
    }
  };

}  // namespace mpl

#endif
