#include "../cactus/index/index.h"
#include "test_utils.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

std::vector<float> random_embedding(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> embedding(dim);
    for (auto& val : embedding) {
        val = dist(gen);
    }
    return embedding;
}

void create_test_index(const std::string& index_path, const std::string& data_path, size_t embedding_dim, uint32_t num_docs = 0) {
    std::ofstream index_file(index_path, std::ios::binary);
    uint32_t magic = cactus::index::MAGIC;
    uint32_t version = cactus::index::VERSION;
    uint32_t embedding_dim_32 = static_cast<uint32_t>(embedding_dim);
    uint32_t num_docs_32 = num_docs;
    index_file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    index_file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    index_file.write(reinterpret_cast<const char*>(&embedding_dim_32), sizeof(uint32_t));
    index_file.write(reinterpret_cast<const char*>(&num_docs_32), sizeof(uint32_t));
    index_file.close();

    std::ofstream data_file(data_path, std::ios::binary);
    data_file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    data_file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    data_file.close();
}

void cleanup_test_files(const std::string& index_path, const std::string& data_path) {
    unlink(index_path.c_str());
    unlink(data_path.c_str());
}

template<typename TestFunc>
bool run_isolated_test(const std::string& test_name, size_t embedding_dim, TestFunc test_func) {
    const std::string index_path = "/tmp/test_" + test_name + ".idx";
    const std::string data_path = "/tmp/test_" + test_name + ".dat";

    cleanup_test_files(index_path, data_path);

    try {
        create_test_index(index_path, data_path, embedding_dim);
        bool result = test_func(index_path, data_path, embedding_dim);
        cleanup_test_files(index_path, data_path);
        return result;
    } catch (...) {
        cleanup_test_files(index_path, data_path);
        return false;
    }
}

bool expect_exception(std::function<void()> func) {
    try {
        func();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool test_constructor_basic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);
        return true;
    };
    return run_isolated_test("constructor_basic", embedding_dim, test);
}

bool test_constructor_missing_index(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        unlink(index_path.c_str());
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_missing_index", embedding_dim, test);
}

bool test_constructor_missing_data(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        unlink(data_path.c_str());
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_missing_data", embedding_dim, test);
}

bool test_constructor_wrong_magic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        std::ofstream index_file(index_path, std::ios::binary);
        uint32_t wrong_magic = 0xDEADBEEF;
        index_file.write(reinterpret_cast<const char*>(&wrong_magic), sizeof(uint32_t));
        index_file.close();

        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_wrong_magic", embedding_dim, test);
}

bool test_constructor_dimension_mismatch(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, uint32_t) {
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, 256);
        });
    };
    return run_isolated_test("constructor_dimension_mismatch", embedding_dim, test);
}

bool test_add_single_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content1", "metadata1"}
        };

        index.add_documents(docs);

        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 && retrieved[0].id == 1;
    };
    return run_isolated_test("add_single_document", embedding_dim, test);
}

bool test_add_multiple_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "metadata" + std::to_string(i)
            });
        }

        index.add_documents(docs);

        auto retrieved = index.get_documents({0, 5, 9});
        return retrieved.size() == 3 &&
               retrieved[0].id == 0 &&
               retrieved[1].id == 5 &&
               retrieved[2].id == 9;
    };
    return run_isolated_test("add_multiple_documents", embedding_dim, test);
}

bool test_add_empty_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);
        std::vector<cactus::index::Document> docs;

        return expect_exception([&]() {
            index.add_documents(docs);
        });
    };
    return run_isolated_test("add_empty_documents", embedding_dim, test);
}

bool test_add_large_content(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::string large_content(1024 * 1024, 'x');
        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), large_content, "metadata1"}
        };

        return expect_exception([&]() {
            index.add_documents(docs);
        });
    };
    return run_isolated_test("add_large_content", embedding_dim, test);
}

bool test_get_single_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto embedding = random_embedding(dim);
        std::vector<cactus::index::Document> docs = {
            {1, embedding, "test content", "test metadata"}
        };
        index.add_documents(docs);

        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 &&
               retrieved[0].id == 1 &&
               retrieved[0].content == "test content" &&
               retrieved[0].metadata == "test metadata" &&
               retrieved[0].embedding.size() == dim;
    };
    return run_isolated_test("get_single_document", embedding_dim, test);
}

bool test_get_multiple_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "meta" + std::to_string(i)
            });
        }
        index.add_documents(docs);

        auto retrieved = index.get_documents({2, 5, 8});
        return retrieved.size() == 3 &&
               retrieved[0].id == 2 &&
               retrieved[0].content == "content2" &&
               retrieved[1].id == 5 &&
               retrieved[1].content == "content5" &&
               retrieved[2].id == 8 &&
               retrieved[2].content == "content8";
    };
    return run_isolated_test("get_multiple_documents", embedding_dim, test);
}

bool test_get_nonexistent_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content", "meta"}
        };
        index.add_documents(docs);

        return expect_exception([&]() {
            index.get_documents({999});
        });
    };
    return run_isolated_test("get_nonexistent_document", embedding_dim, test);
}

bool test_get_empty_doc_ids(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content", "meta"}
        };
        index.add_documents(docs);

        return expect_exception([&]() {
            index.get_documents({});
        });
    };
    return run_isolated_test("get_empty_doc_ids", embedding_dim, test);
}

bool test_get_preserves_order(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 5; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        auto retrieved = index.get_documents({4, 2, 0, 3, 1});
        return retrieved.size() == 5 &&
               retrieved[0].id == 4 &&
               retrieved[1].id == 2 &&
               retrieved[2].id == 0 &&
               retrieved[3].id == 3 &&
               retrieved[4].id == 1;
    };
    return run_isolated_test("get_preserves_order", embedding_dim, test);
}

bool test_get_with_special_characters(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::string special_content = "Content with\nnewlines\t\ttabs and \"quotes\"";
        std::string special_meta = "{\"key\": \"value with spaces\", \"unicode\": \"\\u0041\"}";

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), special_content, special_meta}
        };
        index.add_documents(docs);

        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 &&
               retrieved[0].content == special_content &&
               retrieved[0].metadata == special_meta;
    };
    return run_isolated_test("get_with_special_characters", embedding_dim, test);
}

bool test_get_after_compact(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "meta" + std::to_string(i)
            });
        }
        index.add_documents(docs);

        index.delete_documents({1, 3, 5, 7, 9});
        index.compact();

        auto retrieved = index.get_documents({0, 2, 4, 6, 8});
        return retrieved.size() == 5 &&
               retrieved[0].content == "content0" &&
               retrieved[1].content == "content2" &&
               retrieved[2].content == "content4" &&
               retrieved[3].content == "content6" &&
               retrieved[4].content == "content8";
    };
    return run_isolated_test("get_after_compact", embedding_dim, test);
}

bool test_delete_single_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content1", "metadata1"},
            {2, random_embedding(dim), "content2", "metadata2"}
        };
        index.add_documents(docs);
        index.delete_documents({1});

        return expect_exception([&]() {
            index.get_documents({1});
        });
    };
    return run_isolated_test("delete_single_document", embedding_dim, test);
}

bool test_compact_basic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "metadata" + std::to_string(i)
            });
        }
        index.add_documents(docs);

        struct stat st_before;
        stat(index_path.c_str(), &st_before);
        size_t index_size_before = st_before.st_size;

        index.delete_documents({0, 2, 4, 6, 8});

        index.compact();

        struct stat st_after;
        stat(index_path.c_str(), &st_after);
        size_t index_size_after = st_after.st_size;

        auto retrieved = index.get_documents({1, 3, 5, 7, 9});

        bool size_reduced = index_size_after < index_size_before;
        bool correct_docs = retrieved.size() == 5 &&
                           retrieved[0].id == 1 &&
                           retrieved[1].id == 3 &&
                           retrieved[2].id == 5 &&
                           retrieved[3].id == 7 &&
                           retrieved[4].id == 9;

        bool deleted_docs_gone = expect_exception([&]() {
            index.get_documents({0});
        }) && expect_exception([&]() {
            index.get_documents({2});
        });

        return size_reduced && correct_docs && deleted_docs_gone;
    };
    return run_isolated_test("compact_basic", embedding_dim, test);
}

bool test_compact_query_after(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto embedding1 = random_embedding(dim);
        auto embedding2 = random_embedding(dim);
        auto embedding3 = random_embedding(dim);

        std::vector<cactus::index::Document> docs = {
            {1, embedding1, "content1", "metadata1"},
            {2, embedding2, "content2", "metadata2"},
            {3, embedding3, "content3", "metadata3"}
        };
        index.add_documents(docs);

        index.delete_documents({2});
        index.compact();

        cactus::index::SearchOptions options;
        options.top_k = 2;

        auto results = index.query({embedding1}, options);
        bool query_works = results.size() == 1 && results[0].size() >= 1 && results[0][0].doc_id == 1;

        auto retrieved = index.get_documents({1});
        bool content_preserved = retrieved[0].content == "content1" &&
                                retrieved[0].metadata == "metadata1";

        return query_works && content_preserved;
    };
    return run_isolated_test("compact_query_after", embedding_dim, test);
}

bool test_compact_empty_index(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);
        index.compact();
        return true;
    };
    return run_isolated_test("compact_empty_index", embedding_dim, test);
}

bool test_compact_all_deleted(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 5; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);
        index.delete_documents({0, 1, 2, 3, 4});

        index.compact();

        struct stat st;
        stat(index_path.c_str(), &st);
        size_t index_size = st.st_size;

        size_t header_size = 16;
        bool correct_size = index_size == header_size;

        return correct_size;
    };
    return run_isolated_test("compact_all_deleted", embedding_dim, test);
}

bool test_compact_no_deletions(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 5; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        struct stat st_before;
        stat(index_path.c_str(), &st_before);

        index.compact();

        struct stat st_after;
        stat(index_path.c_str(), &st_after);

        auto retrieved = index.get_documents({0, 1, 2, 3, 4});
        return st_before.st_size == st_after.st_size && retrieved.size() == 5;
    };
    return run_isolated_test("compact_no_deletions", embedding_dim, test);
}

bool test_query_basic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto embedding1 = random_embedding(dim);
        auto embedding2 = random_embedding(dim);

        std::vector<cactus::index::Document> docs = {
            {1, embedding1, "content1", "metadata1"},
            {2, embedding2, "content2", "metadata2"}
        };
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 1;

        auto results = index.query({embedding1}, options);
        return results.size() == 1 && results[0].size() == 1 && results[0][0].doc_id == 1;
    };
    return run_isolated_test("query_basic", embedding_dim, test);
}

bool test_query_topk(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 5;

        auto results = index.query({random_embedding(dim)}, options);
        return results.size() == 1 && results[0].size() <= 5 && results[0].size() > 0;
    };
    return run_isolated_test("query_topk", embedding_dim, test);
}

bool test_stress_many_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 1000; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "meta" + std::to_string(i)
            });
        }

        index.add_documents(docs);

        auto retrieved = index.get_documents({500});
        return retrieved.size() == 1 && retrieved[0].id == 500;
    };
    return run_isolated_test("stress_many_documents", embedding_dim, test);
}

bool test_edge_zero_vector(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<float> zero_embedding(dim, 0.0f);
        std::vector<cactus::index::Document> docs = {
            {1, zero_embedding, "content", "meta"}
        };

        return expect_exception([&]() {
            index.add_documents(docs);
        });
    };
    return run_isolated_test("edge_zero_vector", embedding_dim, test);
}

bool test_edge_duplicate_doc_id(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs1 = {
            {1, random_embedding(dim), "content1", "meta1"}
        };
        index.add_documents(docs1);

        std::vector<cactus::index::Document> docs2 = {
            {1, random_embedding(dim), "content2", "meta2"}
        };

        return expect_exception([&]() {
            index.add_documents(docs2);
        });
    };
    return run_isolated_test("edge_duplicate_doc_id", embedding_dim, test);
}

bool test_edge_negative_doc_id(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {-1, random_embedding(dim), "content", "meta"}
        };

        index.add_documents(docs);
        auto retrieved = index.get_documents({-1});
        return retrieved.size() == 1 && retrieved[0].id == -1;
    };
    return run_isolated_test("edge_negative_doc_id", embedding_dim, test);
}

bool test_edge_empty_content_and_metadata(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "", ""}
        };

        index.add_documents(docs);
        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 &&
               retrieved[0].content == "" &&
               retrieved[0].metadata == "";
    };
    return run_isolated_test("edge_empty_content_and_metadata", embedding_dim, test);
}

bool test_edge_max_content_size(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::string max_content(65535, 'x');
        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), max_content, "meta"}
        };

        index.add_documents(docs);
        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 && retrieved[0].content.size() == 65535;
    };
    return run_isolated_test("edge_max_content_size", embedding_dim, test);
}

bool test_edge_delete_nonexistent_doc(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        return expect_exception([&]() {
            index.delete_documents({999});
        });
    };
    return run_isolated_test("edge_delete_nonexistent_doc", embedding_dim, test);
}

bool test_edge_delete_already_deleted(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content", "meta"}
        };
        index.add_documents(docs);
        index.delete_documents({1});

        return expect_exception([&]() {
            index.delete_documents({1});
        });
    };
    return run_isolated_test("edge_delete_already_deleted", embedding_dim, test);
}

bool test_edge_query_empty_index(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        cactus::index::SearchOptions options;
        options.top_k = 10;

        auto results = index.query({random_embedding(dim)}, options);
        return results.size() == 1 && results[0].empty();
    };
    return run_isolated_test("edge_query_empty_index", embedding_dim, test);
}

bool test_edge_query_with_threshold(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 5; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 10;
        options.score_threshold = 0.99f;

        auto results = index.query({random_embedding(dim)}, options);
        return results.size() == 1 && results[0].size() <= 5;
    };
    return run_isolated_test("edge_query_with_threshold", embedding_dim, test);
}

bool test_edge_batch_query(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 10; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        std::vector<std::vector<float>> queries;
        for (int i = 0; i < 5; ++i) {
            queries.push_back(random_embedding(dim));
        }

        cactus::index::SearchOptions options;
        options.top_k = 3;

        auto results = index.query(queries, options);
        return results.size() == 5 && results[0].size() <= 3;
    };
    return run_isolated_test("edge_batch_query", embedding_dim, test);
}

bool test_reallife_semantic_search(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto doc1_emb = random_embedding(dim);
        auto doc2_emb = random_embedding(dim);
        auto doc3_emb = random_embedding(dim);

        std::vector<cactus::index::Document> docs = {
            {1, doc1_emb, "Machine learning is a subset of artificial intelligence", "{\"category\": \"AI\"}"},
            {2, doc2_emb, "Deep learning uses neural networks with multiple layers", "{\"category\": \"AI\"}"},
            {3, doc3_emb, "Python is a popular programming language", "{\"category\": \"Programming\"}"}
        };
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 2;

        auto results = index.query({doc1_emb}, options);
        bool found_doc1 = false;
        for (const auto& result : results[0]) {
            if (result.doc_id == 1) {
                found_doc1 = true;
                break;
            }
        }
        return results.size() == 1 && found_doc1;
    };
    return run_isolated_test("reallife_semantic_search", embedding_dim, test);
}

bool test_reallife_document_versioning(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "Version 1 content", "{\"version\": 1}"}
        };
        index.add_documents(docs);

        index.delete_documents({1});

        std::vector<cactus::index::Document> updated_docs = {
            {2, random_embedding(dim), "Version 2 content", "{\"version\": 2}"}
        };
        index.add_documents(updated_docs);

        auto retrieved = index.get_documents({2});
        return retrieved.size() == 1 &&
               retrieved[0].content == "Version 2 content" &&
               retrieved[0].metadata == "{\"version\": 2}";
    };
    return run_isolated_test("reallife_document_versioning", embedding_dim, test);
}

bool test_reallife_incremental_indexing(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        for (int batch = 0; batch < 5; ++batch) {
            std::vector<cactus::index::Document> docs;
            for (int i = 0; i < 10; ++i) {
                int doc_id = batch * 10 + i;
                docs.push_back({
                    doc_id,
                    random_embedding(dim),
                    "Batch " + std::to_string(batch) + " doc " + std::to_string(i),
                    "{\"batch\": " + std::to_string(batch) + "}"
                });
            }
            index.add_documents(docs);
        }

        auto retrieved = index.get_documents({0, 25, 49});
        return retrieved.size() == 3 &&
               retrieved[0].id == 0 &&
               retrieved[1].id == 25 &&
               retrieved[2].id == 49;
    };
    return run_isolated_test("reallife_incremental_indexing", embedding_dim, test);
}

bool test_reallife_archive_old_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (int i = 0; i < 20; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "Document " + std::to_string(i),
                "{\"timestamp\": " + std::to_string(i) + "}"
            });
        }
        index.add_documents(docs);

        std::vector<int> old_doc_ids;
        for (int i = 0; i < 10; ++i) {
            old_doc_ids.push_back(i);
        }
        index.delete_documents(old_doc_ids);

        index.compact();

        auto retrieved = index.get_documents({15});
        bool old_deleted = expect_exception([&]() {
            index.get_documents({5});
        });

        return retrieved.size() == 1 && old_deleted;
    };
    return run_isolated_test("reallife_archive_old_documents", embedding_dim, test);
}

bool test_reallife_similarity_deduplication(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto similar_emb = random_embedding(dim);

        std::vector<cactus::index::Document> docs = {
            {1, similar_emb, "Original document", "{\"status\": \"original\"}"},
            {2, similar_emb, "Duplicate document", "{\"status\": \"duplicate\"}"},
            {3, random_embedding(dim), "Different document", "{\"status\": \"unique\"}"}
        };
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 2;

        auto results = index.query({similar_emb}, options);

        bool found_similar = false;
        for (const auto& result : results[0]) {
            if (result.doc_id == 1 || result.doc_id == 2) {
                found_similar = true;
                break;
            }
        }
        return found_similar;
    };
    return run_isolated_test("reallife_similarity_deduplication", embedding_dim, test);
}

bool test_reallife_mixed_operations(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs1;
        for (int i = 0; i < 10; ++i) {
            docs1.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs1);

        index.delete_documents({2, 4, 6});

        std::vector<cactus::index::Document> docs2;
        for (int i = 10; i < 15; ++i) {
            docs2.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs2);

        auto retrieved = index.get_documents({0, 10});

        cactus::index::SearchOptions options;
        options.top_k = 5;
        auto results = index.query({random_embedding(dim)}, options);

        return retrieved.size() == 2 && results.size() == 1;
    };
    return run_isolated_test("reallife_mixed_operations", embedding_dim, test);
}

void run_benchmarks(size_t embedding_dim, uint32_t num_docs) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Index Benchmark Suite                ║\n"
              << "║     Documents: " << std::setw(26) << std::left << num_docs << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    const std::string index_path = "/tmp/bench_index.idx";
    const std::string data_path = "/tmp/bench_data.dat";

    // ========================================
    // INITIALIZATION: Populate index with data
    // ========================================
    std::cout << "\n[INITIALIZATION: Creating and populating index]\n";
    create_test_index(index_path, data_path, embedding_dim);

    auto start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index(index_path, data_path, embedding_dim);
    auto end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "├─ Empty index creation: " << init_duration << "ms\n";

    std::vector<cactus::index::Document> docs;
    docs.reserve(num_docs);
    for (uint32_t i = 0; i < num_docs; ++i) {
        docs.push_back({
            static_cast<int>(i),
            random_embedding(embedding_dim),
            "content_" + std::to_string(i),
            "meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(docs);
    end = std::chrono::high_resolution_clock::now();
    auto populate_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "├─ Populate " << num_docs << " docs: " << populate_duration << "ms\n"
              << "└─ Total initialization: " << (init_duration + populate_duration) << "ms\n";

    // ========================================
    // BENCHMARK: Init (load existing index)
    // ========================================
    std::cout << "\n[BENCHMARK: Init - Load existing index]\n";

    start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index2(index_path, data_path, embedding_dim);
    end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << load_duration << "ms\n";

    // ========================================
    // BENCHMARK: Add (incremental adds)
    // ========================================
    std::cout << "\n[BENCHMARK: Add - Incremental document addition]\n";

    const uint32_t num_adds = std::min(1000u, num_docs / 10);
    std::vector<cactus::index::Document> new_docs;
    new_docs.reserve(num_adds);
    for (uint32_t i = 0; i < num_adds; ++i) {
        new_docs.push_back({
            static_cast<int>(i + num_docs),
            random_embedding(embedding_dim),
            "new_content_" + std::to_string(i),
            "new_meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(new_docs);
    end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << add_duration << "ms\n";

    // ========================================
    // BENCHMARK: Query (similarity search)
    // ========================================
    std::cout << "\n[BENCHMARK: Query - Similarity search]\n";
    cactus::index::SearchOptions options;
    options.top_k = 10;

    auto query_embedding = random_embedding(embedding_dim);

    start = std::chrono::high_resolution_clock::now();
    index.query({query_embedding}, options);
    end = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << query_duration << "ms\n";

    // ========================================
    // BENCHMARK: Get (retrieve by ID)
    // ========================================
    std::cout << "\n[BENCHMARK: Get - Retrieve documents by ID]\n";

    const int num_gets = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> get_doc_ids;
    get_doc_ids.reserve(num_gets);
    for (int i = 0; i < num_gets; ++i) {
        get_doc_ids.push_back(i * (num_docs / num_gets));
    }

    start = std::chrono::high_resolution_clock::now();
    for (const auto& doc_id : get_doc_ids) {
        index.get_documents({doc_id});
    }
    end = std::chrono::high_resolution_clock::now();
    auto get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << get_duration << "ms\n";

    // ========================================
    // BENCHMARK: Delete
    // ========================================
    std::cout << "\n[BENCHMARK: Delete - Remove documents]\n";

    const int num_deletes = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> delete_doc_ids;
    delete_doc_ids.reserve(num_deletes);
    for (int i = 0; i < num_deletes; ++i) {
        delete_doc_ids.push_back(i);
    }

    start = std::chrono::high_resolution_clock::now();
    index.delete_documents(delete_doc_ids);
    end = std::chrono::high_resolution_clock::now();
    auto delete_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << delete_duration << "ms\n";

    // ========================================
    // BENCHMARK: Compact
    // ========================================
    std::cout << "\n[BENCHMARK: Compact - Reclaim deleted space]\n";

    struct stat st_before;
    stat(index_path.c_str(), &st_before);
    size_t index_size_before = st_before.st_size;

    start = std::chrono::high_resolution_clock::now();
    index.compact();
    end = std::chrono::high_resolution_clock::now();
    auto compact_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    struct stat st_after;
    stat(index_path.c_str(), &st_after);
    size_t index_size_after = st_after.st_size;
    size_t space_reclaimed = index_size_before - index_size_after;

    std::cout << "├─ Time: " << compact_duration << "ms\n"
              << "├─ Size before: " << index_size_before << " bytes\n"
              << "├─ Size after:  " << index_size_after << " bytes\n"
              << "└─ Space reclaimed: " << space_reclaimed << " bytes\n";

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Benchmark Summary                    ║\n"
              << "╠══════════════════════════════════════════╣\n"
              << "║ Init:    " << std::setw(29) << std::right << load_duration << "ms ║\n"
              << "║ Add:     " << std::setw(29) << std::right << add_duration << "ms ║\n"
              << "║ Query:   " << std::setw(29) << std::right << query_duration << "ms ║\n"
              << "║ Get:     " << std::setw(29) << std::right << get_duration << "ms ║\n"
              << "║ Delete:  " << std::setw(29) << std::right << delete_duration << "ms ║\n"
              << "║ Compact: " << std::setw(29) << std::right << compact_duration << "ms ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cleanup_test_files(index_path, data_path);
}

int main() {
    const size_t embedding_dim = 1024;
    const uint32_t num_documents = 100000;

    TestUtils::TestRunner runner("Index Tests");

    runner.run_test("constructor_basic", test_constructor_basic(embedding_dim));
    runner.run_test("constructor_missing_index", test_constructor_missing_index(embedding_dim));
    runner.run_test("constructor_missing_data", test_constructor_missing_data(embedding_dim));
    runner.run_test("constructor_wrong_magic", test_constructor_wrong_magic(embedding_dim));
    runner.run_test("constructor_dimension_mismatch", test_constructor_dimension_mismatch(embedding_dim));
    runner.run_test("add_single_document", test_add_single_document(embedding_dim));
    runner.run_test("add_multiple_documents", test_add_multiple_documents(embedding_dim));
    runner.run_test("add_empty_documents", test_add_empty_documents(embedding_dim));
    runner.run_test("add_large_content", test_add_large_content(embedding_dim));
    runner.run_test("get_single_document", test_get_single_document(embedding_dim));
    runner.run_test("get_multiple_documents", test_get_multiple_documents(embedding_dim));
    runner.run_test("get_nonexistent_document", test_get_nonexistent_document(embedding_dim));
    runner.run_test("get_empty_doc_ids", test_get_empty_doc_ids(embedding_dim));
    runner.run_test("get_preserves_order", test_get_preserves_order(embedding_dim));
    runner.run_test("get_with_special_characters", test_get_with_special_characters(embedding_dim));
    runner.run_test("get_after_compact", test_get_after_compact(embedding_dim));
    runner.run_test("delete_single_document", test_delete_single_document(embedding_dim));
    runner.run_test("compact_basic", test_compact_basic(embedding_dim));
    runner.run_test("compact_query_after", test_compact_query_after(embedding_dim));
    runner.run_test("compact_empty_index", test_compact_empty_index(embedding_dim));
    runner.run_test("compact_all_deleted", test_compact_all_deleted(embedding_dim));
    runner.run_test("compact_no_deletions", test_compact_no_deletions(embedding_dim));
    runner.run_test("query_basic", test_query_basic(embedding_dim));
    runner.run_test("query_topk", test_query_topk(embedding_dim));
    runner.run_test("stress_many_documents", test_stress_many_documents(embedding_dim));

    runner.run_test("edge_zero_vector", test_edge_zero_vector(embedding_dim));
    runner.run_test("edge_duplicate_doc_id", test_edge_duplicate_doc_id(embedding_dim));
    runner.run_test("edge_negative_doc_id", test_edge_negative_doc_id(embedding_dim));
    runner.run_test("edge_empty_content_and_metadata", test_edge_empty_content_and_metadata(embedding_dim));
    runner.run_test("edge_max_content_size", test_edge_max_content_size(embedding_dim));
    runner.run_test("edge_delete_nonexistent_doc", test_edge_delete_nonexistent_doc(embedding_dim));
    runner.run_test("edge_delete_already_deleted", test_edge_delete_already_deleted(embedding_dim));
    runner.run_test("edge_query_empty_index", test_edge_query_empty_index(embedding_dim));
    runner.run_test("edge_query_with_threshold", test_edge_query_with_threshold(embedding_dim));
    runner.run_test("edge_batch_query", test_edge_batch_query(embedding_dim));

    runner.run_test("reallife_semantic_search", test_reallife_semantic_search(embedding_dim));
    runner.run_test("reallife_document_versioning", test_reallife_document_versioning(embedding_dim));
    runner.run_test("reallife_incremental_indexing", test_reallife_incremental_indexing(embedding_dim));
    runner.run_test("reallife_archive_old_documents", test_reallife_archive_old_documents(embedding_dim));
    runner.run_test("reallife_similarity_deduplication", test_reallife_similarity_deduplication(embedding_dim));
    runner.run_test("reallife_mixed_operations", test_reallife_mixed_operations(embedding_dim));

    runner.print_summary();

    run_benchmarks(embedding_dim, num_documents);

    return runner.all_passed() ? 0 : 1;
}
