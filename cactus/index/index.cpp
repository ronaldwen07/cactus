#include "index.h"
#include "kernel/kernel.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <numeric>
#include <unordered_set>
#include <queue>
#include <limits>
#include <cerrno>

namespace cactus {
namespace index {

    __fp16 dot_product(const __fp16 *a, const __fp16 *b, size_t dim) {
        __fp16 result;
        cactus_matmul_f16(a, b, &result, 1, dim, 1);
        return result;
    }

    void normalize(__fp16 *v, size_t dim) {
        __fp16 x = dot_product(v, v, dim);
        if (x < 1e-10f) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        cactus_scalar_op_f16(&x, &x, 1, 0, ScalarOpType::SQRT);
        cactus_scalar_op_f16(v, v, dim, x, ScalarOpType::DIVIDE);
    }

    Index::Index(const std::string& index_path, const std::string& data_path, size_t embedding_dim):
        index_path_(index_path), data_path_(data_path), embedding_dim_(embedding_dim),
        index_entry_size_(IndexEntry::size(embedding_dim)),
        index_fd_(-1), data_fd_(-1),
        mapped_index_(nullptr), mapped_data_(nullptr) {
        index_fd_ = open(index_path.c_str(), O_RDWR);
        if (index_fd_ < 0) {
            throw std::runtime_error("Cannot open index file: " + index_path);
        }

        data_fd_ = open(data_path.c_str(), O_RDWR);
        if (data_fd_ < 0) {
            close(index_fd_);
            throw std::runtime_error("Cannot open data file: " + data_path);
        }

        auto cleanup_and_throw = [this](const std::string& msg) {
            close(index_fd_);
            close(data_fd_);
            throw std::runtime_error(msg);
        };

        struct stat index_st, data_st;
        if (fstat(index_fd_, &index_st)) {
            cleanup_and_throw("Cannot get index file size: " + index_path);
        }
        index_file_size_ = index_st.st_size;

        if (fstat(data_fd_, &data_st)) {
            cleanup_and_throw("Cannot get data file size: " + data_path);
        }
        data_file_size_ = data_st.st_size;

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd_, 0);
        if (mapped_index_ == MAP_FAILED) {
            cleanup_and_throw("Cannot map file: " + index_path);
        }

        close(index_fd_);
        index_fd_ = -1;

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, data_fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            munmap(mapped_index_, index_file_size_);
            close(data_fd_);
            throw std::runtime_error("Cannot map file: " + data_path);
        }

        close(data_fd_);
        data_fd_ = -1;

        parse_index_header();
        parse_data_header();

        build_doc_id_map();
    }

    Index::~Index() {
        if (mapped_index_ != nullptr && mapped_index_ != MAP_FAILED) {
            madvise(mapped_index_, index_file_size_, MADV_DONTNEED);
            munmap(mapped_index_, index_file_size_);
        }
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            madvise(mapped_data_, data_file_size_, MADV_DONTNEED);
            munmap(mapped_data_, data_file_size_);
        }
        if (index_fd_ != -1) {
            close(index_fd_);
        }
        if (data_fd_ != -1) {
            close(data_fd_);
        }
    }

    void Index::add_documents(const std::vector<Document>& documents) {
        validate_documents(documents);

        int index_fd = open(index_path_.c_str(), O_RDWR);
        if (index_fd < 0) {
            throw std::runtime_error("Cannot open index file for writing: " + index_path_);
        }

        int data_fd = open(data_path_.c_str(), O_RDWR);
        if (data_fd < 0) {
            close(index_fd);
            throw std::runtime_error("Cannot open data file for writing: " + data_path_);
        }

        auto cleanup_and_throw = [&index_fd, &data_fd](const std::string& msg) {
            close(index_fd);
            close(data_fd);
            throw std::runtime_error(msg);
        };

        off_t data_offset = lseek(data_fd, 0, SEEK_END);

        std::vector<uint64_t> data_offsets;
        data_offsets.reserve(documents.size());

        std::vector<uint32_t> data_sizes;
        data_sizes.reserve(documents.size());

        for (const auto& doc : documents) {
            data_offsets.emplace_back(data_offset);

            const DataEntry entry {
                static_cast<uint16_t>(doc.content.size()),
                static_cast<uint16_t>(doc.metadata.size())
            };

            if (write_full(data_fd, &entry, sizeof(DataEntry)) != static_cast<ssize_t>(sizeof(DataEntry))
                || write_full(data_fd, doc.content.data(), doc.content.size()) != static_cast<ssize_t>(doc.content.size())
                || write_full(data_fd, doc.metadata.data(), doc.metadata.size()) != static_cast<ssize_t>(doc.metadata.size())) {
                cleanup_and_throw("Failed to write document entry");
            }

            uint32_t entry_size = sizeof(DataEntry) + doc.content.size() + doc.metadata.size();
            data_sizes.emplace_back(entry_size);

            if (data_offset + static_cast<off_t>(entry_size) > std::numeric_limits<off_t>::max()) {
                cleanup_and_throw("Data offset overflow when adding documents");
            }

            data_offset += entry_size;
        }

        lseek(index_fd, 0, SEEK_END);

        doc_id_map_.reserve(doc_id_map_.size() + documents.size());

        for (size_t i = 0; i < documents.size(); ++i) {
            const auto& doc = documents[i];

            const IndexEntry entry {
                doc.id,
                data_offsets[i],
                data_sizes[i],
                0
            };

            std::vector<__fp16> normalized_embedding(embedding_dim_);
            cactus_fp32_to_fp16(doc.embedding.data(), normalized_embedding.data(), embedding_dim_);
            normalize(normalized_embedding.data(), embedding_dim_);

            size_t embedding_bytes = embedding_dim_ * sizeof(__fp16);
            if (write_full(index_fd, &entry, sizeof(IndexEntry)) != static_cast<ssize_t>(sizeof(IndexEntry))
                || write_full(index_fd, normalized_embedding.data(), embedding_bytes) != static_cast<ssize_t>(embedding_bytes)) {
                cleanup_and_throw("Failed to write index entry");
            }

            doc_id_map_[doc.id] = num_documents_ + static_cast<uint32_t>(i);
        }

        munmap(mapped_index_, index_file_size_);
        munmap(mapped_data_, data_file_size_);

        index_file_size_ += documents.size() * index_entry_size_;
        data_file_size_ += std::accumulate(data_sizes.begin(), data_sizes.end(), size_t{0});

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd, 0);
        if (mapped_index_ == MAP_FAILED) {
            cleanup_and_throw("Failed to remap index file");
        }

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (mapped_data_ == MAP_FAILED) {
            munmap(mapped_index_, index_file_size_);
            cleanup_and_throw("Failed to remap data file");
        }

        close(index_fd);
        close(data_fd);

        num_documents_ += static_cast<uint32_t>(documents.size());
        IndexHeader* header = reinterpret_cast<IndexHeader*>(mapped_index_);
        header->num_documents = num_documents_;
    }

    void Index::delete_documents(const std::vector<int>& doc_ids) {
        validate_doc_ids(doc_ids);

        char* index_ptr = static_cast<char*>(mapped_index_);
        char* entries = index_ptr + sizeof(IndexHeader);

        if (num_documents_ > 0) {
            size_t last_entry_offset = (num_documents_ - 1) * index_entry_size_;
            if (sizeof(IndexHeader) + last_entry_offset + index_entry_size_ > index_file_size_) {
                throw std::runtime_error("File corrupted: index entry extends beyond file size");
            }
        }

        for (int doc_id : doc_ids) {
            uint32_t i = doc_id_map_.at(doc_id);
            IndexEntry& entry = *reinterpret_cast<IndexEntry*>(entries + i * index_entry_size_);

            entry.flags |= 0x1;
            doc_id_map_.erase(doc_id);
        }
    }

    std::vector<Document> Index::get_documents(const std::vector<int>& doc_ids) {
        validate_doc_ids(doc_ids);

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);
        const char* data_ptr = static_cast<const char*>(mapped_data_);

        if (num_documents_ > 0) {
            size_t last_entry_offset = (num_documents_ - 1) * index_entry_size_;
            if (sizeof(IndexHeader) + last_entry_offset + index_entry_size_ > index_file_size_) {
                throw std::runtime_error("File corrupted: index entry extends beyond file size");
            }
        }

        std::vector<Document> results;
        results.reserve(doc_ids.size());

        for (int doc_id : doc_ids) {
            uint32_t i = doc_id_map_.at(doc_id);
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * index_entry_size_);

            if (static_cast<size_t>(entry.data_offset) + sizeof(DataEntry) > data_file_size_) {
                throw std::runtime_error("File corrupted: data entry extends beyond file size");
            }

            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
            size_t data_entry_size = sizeof(DataEntry) + data_entry->content_len + data_entry->metadata_len;

            if (static_cast<size_t>(entry.data_offset) + data_entry_size > data_file_size_) {
                throw std::runtime_error("File corrupted: data entry extends beyond file size");
            }

            std::vector<float> embedding_f32(embedding_dim_);
            cactus_fp16_to_fp32(entry.embedding(), embedding_f32.data(), embedding_dim_);

            results.emplace_back(
                doc_id,
                std::move(embedding_f32),
                std::string(data_entry->content(), data_entry->content_len),
                std::string(data_entry->metadata(), data_entry->metadata_len)
            );
        }

        return results;
    }

    std::vector<std::vector<SearchResult>> Index::query(const std::vector<std::vector<float>>& embeddings, const SearchOptions& options) {
        if (embeddings.empty()) {
            return {};
        }

        if (num_documents_ > 0) {
            size_t last_entry_offset = (num_documents_ - 1) * index_entry_size_;
            if (sizeof(IndexHeader) + last_entry_offset + index_entry_size_ > index_file_size_) {
                throw std::runtime_error("File corrupted: index entry extends beyond file size");
            }
        }

        std::vector<std::vector<__fp16>> normalized_embeddings;
        normalized_embeddings.reserve(embeddings.size());

        for (const auto& embedding : embeddings) {
            std::vector<__fp16> normalized_embedding(embedding_dim_);
            cactus_fp32_to_fp16(embedding.data(), normalized_embedding.data(), embedding_dim_);
            normalize(normalized_embedding.data(), embedding_dim_);
            normalized_embeddings.emplace_back(std::move(normalized_embedding));
        }

        std::vector<std::vector<SearchResult>> all_results;
        all_results.reserve(embeddings.size());

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);

        auto cmp = [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score;
        };

        for (const auto& normalized_embedding : normalized_embeddings) {
            std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> top_results(cmp);

            for (uint32_t i = 0; i < num_documents_; ++i) {
                const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * index_entry_size_);

                if (entry.flags & 0x1) {
                    continue;
                }

                float score = static_cast<float>(
                    dot_product(normalized_embedding.data(), entry.embedding(), embedding_dim_)
                );

                if (score < options.score_threshold) {
                    continue;
                }

                if (top_results.size() < options.top_k) {
                    top_results.emplace(entry.doc_id, score);
                } else if (score > top_results.top().score) {
                    top_results.pop();
                    top_results.emplace(entry.doc_id, score);
                }
            }

            std::vector<SearchResult> results;
            results.reserve(top_results.size());

            while(!top_results.empty()) {
                results.emplace_back(top_results.top());
                top_results.pop();
            }
            std::reverse(results.begin(), results.end());

            all_results.emplace_back(std::move(results));
        }

        return all_results;
    }

    void Index::compact() {
        std::string temp_index_path = index_path_ + ".tmp";
        std::string temp_data_path = data_path_ + ".tmp";

        int temp_index_fd = open(temp_index_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (temp_index_fd < 0) {
            throw std::runtime_error("Cannot create temporary index file: " + temp_index_path);
        }

        int temp_data_fd = open(temp_data_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (temp_data_fd < 0) {
            close(temp_index_fd);
            unlink(temp_index_path.c_str());
            throw std::runtime_error("Cannot create temporary data file: " + temp_data_path);
        }

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);
        const char* data_ptr = static_cast<const char*>(mapped_data_);

        uint32_t compacted_count = static_cast<uint32_t>(doc_id_map_.size());

        IndexHeader new_header = {
            MAGIC,
            VERSION,
            static_cast<uint32_t>(embedding_dim_),
            compacted_count
        };

        DataHeader new_data_header = {
            MAGIC,
            VERSION
        };

        auto cleanup_and_throw = [&temp_index_fd, &temp_data_fd, &temp_index_path, &temp_data_path](const std::string& msg) {
            close(temp_index_fd);
            close(temp_data_fd);
            unlink(temp_index_path.c_str());
            unlink(temp_data_path.c_str());
            throw std::runtime_error(msg);
        };

        if (write_full(temp_data_fd, &new_data_header, sizeof(DataHeader)) != static_cast<ssize_t>(sizeof(DataHeader))
            || write_full(temp_index_fd, &new_header, sizeof(IndexHeader)) != static_cast<ssize_t>(sizeof(IndexHeader))) {
            cleanup_and_throw("Failed to write headers to temporary files");
        }

        if (num_documents_ > 0) {
            size_t last_entry_offset = (num_documents_ - 1) * index_entry_size_;
            if (sizeof(IndexHeader) + last_entry_offset + index_entry_size_ > index_file_size_) {
                cleanup_and_throw("Compaction failed: File corrupted: index entry extends beyond file size");
            }
        }

        off_t new_data_offset = sizeof(DataHeader);
        std::unordered_map<int, uint32_t> new_doc_id_map;
        new_doc_id_map.reserve(compacted_count);

        uint32_t new_index = 0;
        for (const auto& [doc_id, index] : doc_id_map_) {
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + index * index_entry_size_);

            if (static_cast<size_t>(entry.data_offset) + sizeof(DataEntry) > data_file_size_) {
                cleanup_and_throw("Compaction failed: File corrupted: data entry extends beyond file size");
            }

            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
            uint32_t data_entry_size = sizeof(DataEntry) + data_entry->content_len + data_entry->metadata_len;

            if (static_cast<size_t>(entry.data_offset) + data_entry_size > data_file_size_) {
                cleanup_and_throw("Compaction failed: File corrupted: data entry extends beyond file size");
            }

            IndexEntry new_entry = {
                entry.doc_id,
                static_cast<uint64_t>(new_data_offset),
                data_entry_size,
                0
            };

            size_t embedding_bytes = embedding_dim_ * sizeof(__fp16);
            if (write_full(temp_data_fd, data_entry, data_entry_size) != static_cast<ssize_t>(data_entry_size)
                || write_full(temp_index_fd, &new_entry, sizeof(IndexEntry)) != static_cast<ssize_t>(sizeof(IndexEntry))
                || write_full(temp_index_fd, entry.embedding(), embedding_bytes) != static_cast<ssize_t>(embedding_bytes)) {
                cleanup_and_throw("Failed to write data entry during compaction");
            }

            new_doc_id_map[doc_id] = new_index;
            ++new_index;
            new_data_offset += data_entry_size;
        }

        if (fsync(temp_index_fd) != 0 || fsync(temp_data_fd) != 0) {
            cleanup_and_throw("Failed to sync temporary files");
        }

        close(temp_index_fd);
        close(temp_data_fd);

        munmap(mapped_index_, index_file_size_);
        munmap(mapped_data_, data_file_size_);

        std::string backup_index_path = index_path_ + ".backup";
        std::string backup_data_path = data_path_ + ".backup";

        if (rename(index_path_.c_str(), backup_index_path.c_str()) != 0
            || rename(data_path_.c_str(), backup_data_path.c_str()) != 0
            || rename(temp_index_path.c_str(), index_path_.c_str()) != 0
            || rename(temp_data_path.c_str(), data_path_.c_str()) != 0) {
            rename(backup_index_path.c_str(), index_path_.c_str());
            rename(backup_data_path.c_str(), data_path_.c_str());
            unlink(temp_index_path.c_str());
            unlink(temp_data_path.c_str());
            throw std::runtime_error("Failed to replace files during compaction");
        }

        unlink(backup_index_path.c_str());
        unlink(backup_data_path.c_str());

        int new_index_fd = open(index_path_.c_str(), O_RDWR);
        if (new_index_fd < 0) {
            throw std::runtime_error("Cannot open new index file: " + index_path_);
        }

        int new_data_fd = open(data_path_.c_str(), O_RDWR);
        if (new_data_fd < 0) {
            close(new_index_fd);
            throw std::runtime_error("Cannot open new data file: " + data_path_);
        }

        auto cleanup_fds = [&new_index_fd, &new_data_fd](const std::string& msg) {
            close(new_index_fd);
            close(new_data_fd);
            throw std::runtime_error(msg);
        };

        struct stat index_st, data_st;
        if (fstat(new_index_fd, &index_st)) {
            cleanup_fds("Cannot get index file size: " + index_path_);
        }
        index_file_size_ = index_st.st_size;

        if (fstat(new_data_fd, &data_st)) {
            cleanup_fds("Cannot get data file size: " + data_path_);
        }
        data_file_size_ = data_st.st_size;

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, new_index_fd, 0);
        if (mapped_index_ == MAP_FAILED) {
            cleanup_fds("Cannot map file: " + index_path_);
        }

        close(new_index_fd);

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, new_data_fd, 0);
        if (mapped_data_ == MAP_FAILED) {
            munmap(mapped_index_, index_file_size_);
            close(new_data_fd);
            throw std::runtime_error("Cannot map file: " + data_path_);
        }

        close(new_data_fd);

        num_documents_ = compacted_count;
        doc_id_map_ = std::move(new_doc_id_map);
    }

    void Index::parse_index_header() {
        const char* index_ptr = static_cast<const char*>(mapped_index_);
        size_t offset = 0;

        if (index_file_size_ < sizeof(IndexHeader)) {
            throw std::runtime_error("Index file too small: insufficient data for header");
        }

        IndexHeader header;
        header.magic = *reinterpret_cast<const decltype(header.magic)*>(index_ptr + offset);
        offset += sizeof(header.magic);

        if (header.magic != MAGIC) {
            throw std::runtime_error("Invalid index file magic number");
        }

        header.version = *reinterpret_cast<const decltype(header.version)*>(index_ptr + offset);
        offset += sizeof(header.version);

        if (header.version != VERSION) {
            throw std::runtime_error("Index file version mismatch");
        }

        header.embedding_dim = *reinterpret_cast<const decltype(header.embedding_dim)*>(index_ptr + offset);
        offset += sizeof(header.embedding_dim);

        if (static_cast<size_t>(header.embedding_dim) != embedding_dim_) {
            throw std::runtime_error("Embedding dimension mismatch");
        }

        num_documents_ = *reinterpret_cast<const decltype(header.num_documents)*>(index_ptr + offset);
    }

    void Index::parse_data_header() {
        const char* data_ptr = static_cast<const char*>(mapped_data_);
        size_t offset = 0;

        if (data_file_size_ < sizeof(DataHeader)) {
            throw std::runtime_error("Data file too small: insufficient data for header");
        }

        DataHeader header;
        header.magic = *reinterpret_cast<const decltype(header.magic)*>(data_ptr + offset);
        offset += sizeof(header.magic);

        if (header.magic != MAGIC) {
            throw std::runtime_error("Invalid data file magic number");
        }

        header.version = *reinterpret_cast<const decltype(header.version)*>(data_ptr + offset);
        offset += sizeof(header.version);

        if (header.version != VERSION) {
            throw std::runtime_error("Data file version mismatch");
        }
    }

    void Index::build_doc_id_map() {
        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);

        if (num_documents_ > 0) {
            size_t last_entry_offset = (num_documents_ - 1) * index_entry_size_;
            if (sizeof(IndexHeader) + last_entry_offset + index_entry_size_ > index_file_size_) {
                throw std::runtime_error("File corrupted: index entry extends beyond file size");
            }
        }

        doc_id_map_.reserve(static_cast<size_t>(num_documents_));

        for (uint32_t i = 0; i < num_documents_; ++i) {
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * index_entry_size_);

            if (entry.flags & 0x1) {
                continue;
            }

            doc_id_map_[entry.doc_id] = i;
        }
    }

    void Index::validate_documents(const std::vector<Document>& documents) {
        if (documents.empty()) {
            throw std::runtime_error("Documents vector is empty");
        }

        std::unordered_set<int> seen_ids;
        for (const auto& doc : documents) {
            if (doc.content.size() > 65535) {
                throw std::runtime_error("Document content too long (max 65535 bytes) for ID: " + std::to_string(doc.id));
            }

            if (doc.metadata.size() > 65535) {
                throw std::runtime_error("Document metadata too long (max 65535 bytes) for ID: " + std::to_string(doc.id));
            }

            if (doc.embedding.size() != embedding_dim_) {
                throw std::runtime_error("Document embedding dimension mismatch for ID: " + std::to_string(doc.id));
            }

            if (seen_ids.find(doc.id) != seen_ids.end()) {
                throw std::runtime_error("Duplicate document ID in input: " + std::to_string(doc.id));
            }
            seen_ids.insert(doc.id);

            if (doc_id_map_.find(doc.id) != doc_id_map_.end()) {
                throw std::runtime_error("Document ID already exists: " + std::to_string(doc.id));
            }
        }
    }

    void Index::validate_doc_ids(const std::vector<int>& doc_ids) {
        if (doc_ids.empty()) {
            throw std::runtime_error("Document IDs vector is empty");
        }

        std::unordered_set<int> seen_ids;
        for (int doc_id : doc_ids) {
            if (seen_ids.find(doc_id) != seen_ids.end()) {
                throw std::runtime_error("Duplicate document ID in input: " + std::to_string(doc_id));
            }
            seen_ids.insert(doc_id);

            if (doc_id_map_.find(doc_id) == doc_id_map_.end()) {
                throw std::runtime_error("Document ID not found: " + std::to_string(doc_id));
            }
        }
    }

    ssize_t Index::write_full(int fd, const void* buf, size_t count) {
        size_t written = 0;
        const char* ptr = static_cast<const char*>(buf);
        while (written < count) {
            ssize_t n = write(fd, ptr + written, count - written);
            if (n < 0) {
                if (errno == EINTR) continue;
                return -1;
            }
            if (n == 0) return written;
            written += n;
        }
        return written;
    }

} // namespace index
} // namespace cactus