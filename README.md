<img src="assets/banner.jpg" alt="Logo" style="border-radius: 30px; width: 100%;">

Cross-platform & energy-efficient kernels, runtime and AI inference engine for mobile devices. 

# Cactus Graph 
Cactus Graph is a general numerical computing framework for implementing 
any model, like PyTorch for mobile devices.

```cpp
#include cactus.h

CactusGraph graph;
auto a = graph.input({2, 3}, Precision::FP16);
auto b = graph.input({3, 4}, Precision::INT8);

auto x1 = graph.matmul(a, b, false);
auto x2 = graph.transpose(x1);
auto result = graph.matmul(b, x2, true);

float a_data[6] = {1.1f, 2.3f, 3.4f, 4.2f, 5.7f, 6.8f};
float b_data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
graph.set_input(a, a_data, Precision::FP16);
graph.set_input(b, b_data, Precision::INT8);

graph.execute();
void* output_data = graph.get_output(result);

graph.hard_reset(); 

```

# Cactus Engine
Cactus Engine is an AI inference engine with OpenAI-compatible APIs built on top of Cactus Graphs.

```cpp
#include cactus.h

cactus_set_pro_key(""); // email founders@cactuscompute.com for optional key

cactus_model_t model = cactus_init("path/to/weight/folder", 2048);

const char* messages = R"([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Henry Ndubuaku"}
])";

const char* options = R"({
    "max_tokens": 50,
    "stop_sequences": ["<|im_end|>"]
})";

char response[1024];
int result = cactus_complete(model, messages, response, sizeof(response), options, nullptr, nullptr, nullptr);
```
Example response from Gemma3-270m-INT8
```json
{
    "success": true,
    "response": "Hi there! I'm just a friendly assistant.",
    "time_to_first_token_ms": 45.23,
    "total_time_ms": 163.67,
    "tokens_per_second": 168.42,
    "prefill_tokens": 28,
    "decode_tokens": 50,
    "total_tokens": 78
}
```

# Performance

- <sub>**Models:** LFM2-VL-450m & Whisper-Small</sub>
- <sub>**Decode** = toks/sec, **P/D** = prefill/decode, **VLM** = 256×256 image, **STT** = 30s audio</sub>
- <sub>**Cactus Pro**: Uses NPU for realtime and large context (Apple for now), scores are marked with *</sub>

| Device | Short Decode | 4k-P/D | VLM-TTFT | VLM-Dec | STT-TTFT | STT-Dec |
|--------|--------|--------|----------|---------|----------|---------|
| Mac M4 Pro | 170 | 989/100 | 0.2s/0.1s* | 168 | 0.9s/0.2s* | 92 |
| Mac M3 Pro | 140 | 890/93 | 0.3s/0.1s* | 149 | 1.5s/0.4s* | 81 |
| iPad/Mac M4 | 134 | 603/106 | 0.3s/0.1s* | 129 | 1.8s0.3s* | 70 |
| iPad/Mac M3 | 117 | 525/93 | 0.4s/0.1s* | 111 | 2.8s/0.7s* | 61 |
| iPhone 17 Pro | 126 | 428/84 | 0.5s/0.1s* | 120 | 3.0s/0.6s* | 80 |
| iPhone 16 Pro | 106 | 380/81 | 0.6s/0.2s* | 101 | 4.3s/0.7s* | 75 |
| iPhone 15 Pro | 90 | 330/75 | 0.7s/0.3s* | 92 | 4.5s/0.8s* | 70 |
| Galaxy S25 Ultra | 80 | 355/52 | 0.7s | 70 | 3.6s/- | 32 |
| Nothing 3 | 56 | 320/46 | 0.8s | 54 | 4.5s | 55 |
| Pixel 6a | 25 | 108/24 | 2.3s | 25 | 9.6 | 15 |
| Raspberry Pi 5 | 20 | 292/18 | 1.7s | 23 | 15s | 16 |


# Supported models

| Model | Zipped INT4/INT8 | RAM@4k-Context | Completion | Tool Call | Vision | Embed | Speech | Pro |
|-------|-----------|--------|------------|-----------|--------|-------|--------|-----|
| google/gemma-3-270m-it | 115MB/172MB | 180MB | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| google/functiongemma-270m-it | 115MB/172MB | 180MB | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-small | 104MB/282MB | 334MB | ✗ | ✗ | ✗ | ✓ | ✓ | Apple |
| LiquidAI/LFM2-350M | 153MB/233MB | 374MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| HuggingFaceTB/SmolLM2-360m-Instruct | 140MB/227MB | 374MB | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| LiquidAI/LFM2-VL-450M | 318MB/480MB | 445MB | ✓ | ✗ | ✓ | ✓ | ✗ | Apple |
| nomic-ai/nomic-embed-text-v2-moe | 211MB/456MB | 529MB | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Qwen/Qwen3-0.6B | 234MB/394MB | 643MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Qwen/Qwen3-Embedding-0.6B | 234MB/394MB | 643MB | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-700M | 300MB/467MB | 720MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| google/gemma-3-1b-it | 320MB/642MB | 1080MB | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| openai/whisper-medium | 320MB/646MB | 1180MB | ✗ | ✗ | ✗ | ✓ | ✓ | Apple |
| LiquidAI/LFM2.5-1.2B-Instruct | 474MB/722MB | 1280MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-1.2B-RAG | 474MB/722MB | 1280MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2-1.2B-Tool | 474MB/722MB | 1280MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| LiquidAI/LFM2.5-VL-1.6B | 954MB/1440MB | 1280MB | ✓ | ✗ | ✓ | ✓ | ✗ | Apple |
| Qwen/Qwen3-1.7B | 801MB/1161MB | 1680MB | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 801MB/1161MB | 1680MB | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |


# Using this repo on a Mac

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus && source ./setup
```

- <sub> `[model]` is a HuggingFace name from the table above (default: `google/gemma-3-270m-it`)</sub>
- <sub> Common flags: `--precision INT4|INT8|FP16` (default: INT4), `--token <hf_token>`</sub>
- <sub>Always run `source ./setup` in any new terminal.</sub>

| Command | Description |
|---------|-------------|
| `cactus run [model]` | Opens playground (auto downloads model) |
| `cactus download [model]` | Downloads model to `./weights` |
| `cactus convert [model] [dir]` | Converts model, supports LoRA merging (`--lora <path>`) |
| `cactus build` | Builds for ARM (`--apple` or `--android`) |
| `cactus test` | Runs tests (`--ios` / `--android` for device testing) |
| `cactus clean` | Removes build artifacts |
| `cactus --help` | Shows all commands and flags |

# Python Package

Cactus python package is auto installed for researchers and testing.

```python
from cactus_ffi import cactus_init, cactus_complete, cactus_destroy

model = cactus_init("weights/lfm2-vl-450m", context_size=2048)

messages = json.dumps([{"role": "user", "content": "What is 2+2?"}])
response = cactus_complete(model, messages) # returns JSON

cactus_destroy(model)
```

Setup and full example:
```bash
cactus build
cactus download LiquidAI/LFM2-VL-450M
python tools/example.py
```

# Using in your apps

- [Kotlin Multiplatform SDK](https://github.com/cactus-compute/cactus-kotlin)
- [Flutter SDK](https://github.com/cactus-compute/cactus-flutter)
- [React Native SDK](https://github.com/cactus-compute/cactus-react-native)
- [Swift SDK](https://github.com/mhayes853/swift-cactus)
- [Rust SDK](https://github.com/mrsarac/cactus-rs)

# Try demo apps

- [iOS Demo](https://apps.apple.com/gb/app/cactus-chat/id6744444212)
- [Android Demo](https://play.google.com/store/apps/details?id=com.rshemetsubuser.myapp)
