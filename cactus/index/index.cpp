#include "index.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <queue>

namespace cactus {
namespace index {

    float dot_product(const float *a, const float *b, size_t dim) {
        float result = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    void normalize(float *v, size_t dim) {
        float norm = std::sqrt(dot_product(v, v, dim));

        if (norm < 1e-6f) {
            return;
        }

        for (size_t i = 0; i < dim; ++i) {
            v[i] /= norm;
        }
    }

    Index::Index(const std::string& index_path, const std::string& data_path, uint32_t embedding_dim):
        index_path_(index_path), data_path_(data_path), embedding_dim_(embedding_dim) {
        index_fd_ = open(index_path.c_str(), O_RDONLY);
        data_fd_ = open(data_path.c_str(), O_RDONLY);

        struct stat index_st, data_st;
        if (fstat(index_fd_, &index_st)) {
            close(index_fd_);
            throw std::runtime_error("Cannot get index file size: " + index_path);
        }
        index_file_size_ = index_st.st_size;

        if (fstat(data_fd_, &data_st)) {
            close(data_fd_);
            throw std::runtime_error("Cannot get data file size: " + data_path);
        }
        data_file_size_ = data_st.st_size;

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd_, 0);
        if (mapped_index_ == MAP_FAILED) {
            close(index_fd_);
            throw std::runtime_error("Cannot map file: " + index_path);
        }

        close(index_fd_);
        index_fd_ = -1;

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, data_fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            close(data_fd_);
            throw std::runtime_error("Cannot map file: " + data_path);
        }

        close(data_fd_);
        data_fd_ = -1;

        parse_index_header();
        parse_data_header();
    }

    void Index::add_documents(const std::vector<Document>& documents) {
        // Implementation goes here
    }

    void Index::delete_documents(const std::vector<std::string>& doc_ids) {
        std::unordered_set<std::string> doc_id_set(doc_ids.begin(), doc_ids.end());

        char* index_ptr = static_cast<char*>(mapped_index_);
        char* entries = index_ptr + sizeof(IndexHeader);

        const char* data_ptr = static_cast<const char*>(mapped_data_);

        for (uint32_t i = 0; i < num_documents_; ++i) {
            IndexEntry& entry = *reinterpret_cast<IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));

            if (entry.flags & 0x1) {
                continue;
            }

            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
            const std::string doc_id = std::string(data_entry->doc_id(), data_entry->doc_id_len);

            if (doc_id_set.find(doc_id) != doc_id_set.end()) {
                entry.flags |= 0x1;
            }
        }
    }

    std::vector<Document> Index::get_documents(const std::vector<std::string>& doc_ids) {
        std::unordered_map<std::string, std::vector<size_t>> doc_id_map;
        for (size_t i = 0; i < doc_ids.size(); ++i) {
            doc_id_map[doc_ids[i]].emplace_back(i);
        }

        std::vector<Document> results(doc_ids.size());

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);

        const char* data_ptr = static_cast<const char*>(mapped_data_);

        uint32_t found_count = 0;
        for (uint32_t i = 0; i < num_documents_; ++i) {
            if (found_count == doc_ids.size()) {
                break;
            }

            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));

            if (entry.flags & 0x1) {
                continue;
            }

            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
            const std::string doc_id = std::string(data_entry->doc_id(), data_entry->doc_id_len);

            if (doc_id_map.find(doc_id) != doc_id_map.end()) {
                const float* entry_embedding = entry.embedding();
                const Document doc{
                    doc_id,
                    std::vector<float>(entry_embedding, entry_embedding + embedding_dim_),
                    std::string(data_entry->content(), data_entry->content_len),
                    std::string(data_entry->metadata(), data_entry->metadata_len)
                };
                for (size_t idx : doc_id_map[doc_id]) {
                    results[idx] = doc;
                }
                found_count += doc_id_map[doc_id].size();
            }
        }

        if (found_count != doc_ids.size()) {
            for (size_t i = 0; i < doc_ids.size(); ++i) {
                if (results[i].id.empty()) {
                    throw std::runtime_error("Document ID not found: " + doc_ids[i]);
                }
            }
        }

        return results;
    }

    std::vector<SearchResult> Index::query(const std::vector<float>& embedding, const SearchOptions& options) {
        if (options.top_k == 0) {
            return {};
        }

        std::vector<float> normalized_embedding(embedding);
        normalize(normalized_embedding.data(), embedding_dim_);

        auto cmp = [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score; 
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> top_results(cmp);

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);

        const char* data_ptr = static_cast<const char*>(mapped_data_);

        for (uint32_t i = 0; i < num_documents_; ++i) {
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));
            const float* entry_embedding = entry.embedding();

            if (entry.flags & 0x1) {
                continue;
            }

            float score = dot_product(normalized_embedding.data(), entry_embedding, embedding_dim_);

            if (score < options.score_threshold) {
                continue;
            }

            if (top_results.size() < options.top_k) {
                const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
                top_results.emplace(std::string(data_entry->doc_id(), data_entry->doc_id_len), score);
            } else if (score > top_results.top().score) {
                top_results.pop();
                const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
                top_results.emplace(std::string(data_entry->doc_id(), data_entry->doc_id_len), score);
            }
        }

        std::vector<SearchResult> results(top_results.size());
        for (size_t i = top_results.size(); i-- > 0;) {
            results[i] = top_results.top();
            top_results.pop();
        }

        return results;
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

        if (header.embedding_dim != embedding_dim_) {
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

} // namespace index
} // namespace cactus