#include "test_utils.h"
#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>

using namespace EngineTestUtils;

const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
const char* g_whisper_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";

// Configurable chunk duration in seconds
constexpr int CHUNK_DURATION_SECONDS = 3;

// Signal handling for graceful shutdown
std::atomic<bool> g_keep_running(true);

void signal_handler(int signum) {
    (void)signum;
    std::cout << "\n\n[!] Interrupt signal received. Stopping...\n";
    g_keep_running = false;
}

const std::vector<std::string> SPECIAL_TOKENS = {
    "<|startoftranscript|>", "<|endoftranscript|>",
    "<|en|>", "<|transcribe|>", "<|notimestamps|>",
    "<|startoflm|>", "<|startofprev|>",
    "[no audio]", "[BLANK_AUDIO]", "(static)",
    "[silence]", "[background noise]"
};

// Custom streaming callback that filters special tokens
void filtered_stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token ? token : "");
    data->token_ids.push_back(token_id);
    data->token_count++;

    std::string out = token ? token : "";

    bool is_special = false;
    for (const auto& special : SPECIAL_TOKENS) {
        if (out.find(special) != std::string::npos) {
            is_special = true;
            break;
        }
    }

    if (!is_special && !out.empty()) {
        for (char& c : out) if (c == '\n') c = ' ';
        std::cout << out << std::flush;
    }
}

int main() {
    std::cout << "\nSTREAMING MICROPHONE TRANSCRIPTION (3-second chunks)\n\n";

#ifndef HAVE_SDL2
    std::cerr << "[✗] This test requires SDL2 for microphone input\n";
    std::cerr << "    Please install SDL2 and rebuild with SDL2 support\n";
    return 1;
#else

    if (!g_transcribe_model_path) {
        std::cerr << "[✗] CACTUS_TEST_TRANSCRIBE_MODEL environment variable not set\n";
        std::cerr << "    Please set it to your whisper model path\n";
        return 1;
    }

    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize whisper model
    std::cout << "[*] Initializing Whisper model...\n";
    cactus_model_t model = cactus_init(g_transcribe_model_path, 2048, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return 1;
    }
    std::cout << "[✓] Model initialized successfully\n\n";

    // Initialize audio capture
    const int sample_rate = 16000;  // 16kHz required for Whisper
    const int chunk_duration_ms = CHUNK_DURATION_SECONDS * 1000;

    // Buffer needs to be larger than chunk size to avoid overflow
    const int buffer_size_ms = chunk_duration_ms * 4;  // 4x chunk size for safety

    // Initialize SDL audio subsystem
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        std::cerr << "[✗] Failed to initialize SDL: " << SDL_GetError() << "\n";
        cactus_destroy(model);
        return 1;
    }

    // List available audio devices
    std::cout << "\n[*] Available audio capture devices:\n";
    int num_devices = SDL_GetNumAudioDevices(1);  // 1 = recording devices
    if (num_devices < 1) {
        std::cerr << "[✗] No audio capture devices found\n";
        SDL_Quit();
        cactus_destroy(model);
        return 1;
    }

    for (int i = 0; i < num_devices; i++) {
        std::cout << "  [" << i << "] " << SDL_GetAudioDeviceName(i, 1) << "\n";
    }

    // Ask user to select device
    std::cout << "\nSelect audio device (0-" << (num_devices - 1) << "): ";
    int selected_device = 0;
    std::cin >> selected_device;

    if (selected_device < 0 || selected_device >= num_devices) {
        std::cerr << "[✗] Invalid device selection\n";
        SDL_Quit();
        cactus_destroy(model);
        return 1;
    }

    std::cout << "\n[*] Initializing device: " << SDL_GetAudioDeviceName(selected_device, 1) << "\n";
    AudioCapture audio_capture(buffer_size_ms);
    if (!audio_capture.init(selected_device, sample_rate)) {
        std::cerr << "[✗] Failed to initialize audio capture\n";
        std::cerr << "    Make sure the selected device supports recording\n";
        SDL_Quit();
        cactus_destroy(model);
        return 1;
    }
    std::cout << "[✓] Microphone initialized (16kHz)\n\n";

    audio_capture.resume();

    std::cout << "\n🎤 RECORDING STARTED - Speak now... (Press Ctrl+C to stop)\n\n";

    std::string full_transcript;
    int chunk_number = 0;

    while (g_keep_running) {
        // Wait for chunk duration to accumulate audio
        std::this_thread::sleep_for(std::chrono::milliseconds(chunk_duration_ms));

        if (!g_keep_running) break;

        chunk_number++;

        // Get all available audio from the buffer
        std::vector<float> audio_float;
        size_t num_samples = audio_capture.get_all(audio_float);

        // Clear the buffer for the next chunk
        audio_capture.clear();

        if (num_samples == 0) {
            continue;  // Silent chunk, skip
        }

        // Convert float32 [-1, 1] to int16_t PCM
        std::vector<int16_t> pcm_samples(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            float clamped = std::max(-1.0f, std::min(1.0f, audio_float[i]));
            pcm_samples[i] = static_cast<int16_t>(clamped * 32767.0f);
        }

        // Transcribe this chunk
        char response[1 << 15] = {0};
        StreamingData stream;
        stream.model = model;

        int rc = cactus_transcribe(
            model,
            nullptr,  // No file path, using PCM buffer
            g_whisper_prompt,
            response,
            sizeof(response),
            R"({"max_tokens": 100})",
            filtered_stream_callback,
            &stream,
            reinterpret_cast<const uint8_t*>(pcm_samples.data()),
            pcm_samples.size() * sizeof(int16_t)
        );

        if (rc <= 0) {
            // Reset even on failure to clear any partial state
            cactus_reset(model);
            continue;
        }

        std::string chunk_text;
        for (const auto& token : stream.tokens) {
            chunk_text += token;
        }

        for (const auto& special : SPECIAL_TOKENS) {
            size_t pos;
            while ((pos = chunk_text.find(special)) != std::string::npos) {
                chunk_text.erase(pos, special.length());
            }
        }

        size_t start = chunk_text.find_first_not_of(" \n");
        size_t end = chunk_text.find_last_not_of(" \n");
        if (start != std::string::npos) {
            chunk_text = chunk_text.substr(start, end - start + 1);
        } else {
            chunk_text.clear();
        }

        if (!chunk_text.empty()) {
            // Add space before appending if needed
            if (!full_transcript.empty() && full_transcript.back() != ' ') {
                full_transcript += " ";
            }
            full_transcript += chunk_text;
        }

        // Reset model state AFTER processing results to avoid interrupting the current transcription
        cactus_reset(model);
    }

    audio_capture.pause();
    std::cout << "\n\nRECORDING STOPPED\n\n";

    if (!full_transcript.empty()) {
        std::cout << "[*] Chunks: " << chunk_number << " (~" << (chunk_number * CHUNK_DURATION_SECONDS) << "s)\n";
    } else {
        std::cout << "[!] No speech transcribed\n";
    }

    cactus_destroy(model);
    std::cout << "[✓] Test completed\n";
    return 0;

#endif  // HAVE_SDL2
}
