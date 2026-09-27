<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/naive-n0.5-flash-logo-dark.svg">
    <img src="assets/naive-n0.5-flash-logo.svg" width="480" alt="Naive-N0.5-Flash">
  </picture>
</div>

<p align="center"><strong>Building Frontier AI with AI</strong></p>

<div align="center">

[🏠 Homepage][website] · [📰 Technical Blog][blog] · [💻 GitHub][github] · [🤗 Hugging Face][weights]

</div>

## Introduction

Naive-N0.5-Flash is an open-weight **309B MoE model with 15.5B active parameters**, built for **coding and AI R&D**. It supports a **native 1M-token context window** through a hybrid of Sliding-Window Attention (SWA) and lightweight DeepSeek Sparse Attention (DSA), with no full-attention layers.

### Key Features

- **Native 1M context, without full attention.** Naive-N0.5-Flash combines Sliding-Window Attention (SWA) and lightweight DeepSeek Sparse Attention (DSA) with GQA4 at a predominantly 5:1 SWA–DSA layout. The entire network remains local or sparse, with no full-attention layers.
- **AI-optimized inference up to 2,000 tokens/s.** NaiveRT, our inference system for Naive-N0.5-Flash, was built and optimized through AI-centered R&D. It combines mega-kernel fusion, Programmatic Dependent Launch (PDL), and speculative decoding, delivering 50 tokens/s per user in Standard mode and up to 2,000 tokens/s in Ultrafast mode. See the NaiveRT case study in the [technical blog][blog] for the implementation and optimization process.
- **Open weights and API.** Model weights and inference code are released under the MIT license. API access will also be provided, with pricing set at $0.10 / $0.40 / $0.01 per million tokens for input, output, and cache reads, respectively.

## Model Architecture

<table align="center">
  <thead>
    <tr>
      <th>Property</th>
      <th>Specification</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Architecture</td><td>Mixture-of-Experts (MoE)</td></tr>
    <tr><td>Total parameters</td><td>309B</td></tr>
    <tr><td>Active parameters</td><td>15.5B</td></tr>
    <tr><td>Context length</td><td>Native 1M tokens</td></tr>
    <tr><td>Transformer layers</td><td>48</td></tr>
    <tr><td>Attention-layer composition</td><td>39 SWA layers + 9 DSA layers</td></tr>
    <tr><td>Attention mechanism</td><td>Hybrid SWA–DSA</td></tr>
    <tr><td>SWA window</td><td>128 tokens</td></tr>
    <tr><td>DSA token selection</td><td>Top 2,048 tokens for backbone attention</td></tr>
    <tr><td>DSA KV groups</td><td>4 (GQA4)</td></tr>
    <tr><td>Indexer query heads</td><td>16</td></tr>
  </tbody>
</table>

### Hybrid SWA–DSA Attention

Naive-N0.5-Flash builds on the open-weight MiMo-V2.5 base model, which has a simple architecture with strong foundational capabilities in world knowledge and deep research. Most layers use Sliding-Window Attention (SWA), whose per-token decoding cost does not grow with context length, while a small number of global-attention layers preserve long-range information. At million-token context lengths, however, these global-attention layers account for much of the decoding overhead.

Naive-N0.5-Flash replaces the global-attention layers with DeepSeek Sparse Attention (DSA). A lightweight indexer scores the full history, while the backbone computes attention only over a selected subset of tokens. Although the indexer still scans the full history and the full KV cache is retained, sparse attention substantially reduces attention computation and memory access. Adapting the model to this new attention structure was one objective of continued pretraining.

![Hybrid SWA–DSA architecture showing the network stack and the DSA attention module, including the 16-head indexer and top-2,048 token selection.][architecture-figure]

<p align="center"><em>Figure 1. The hybrid attention stack and DSA module.</em></p>

The network consists of **eight six-layer modules**. A standard module contains five SWA layers followed by one DSA layer, with the first layer of the first module also replaced by DSA. SWA uses a **128-token window**, while DSA selects the **top 2,048 tokens** for backbone attention. Both attention types incorporate sink bias.

Unlike the original MLA-based DSA implementation, Naive-N0.5-Flash replaces MLA with grouped-query attention (GQA) using **four KV groups**. For the architecture design process and indexer efficiency comparison, see model architecture in the [technical blog][blog].

### Training Overview

Following the architectural changes, Naive-N0.5-Flash completed 3.25T tokens of multi-stage training with a native 1M-token context window: 50B tokens of Indexer Warmup, 3T tokens of Sparse Attention Training, and 200B tokens of Learning Rate Decay. This process adapted the model to its new sparse attention architecture while substantially improving its AI R&D and coding capabilities. See the [technical blog][blog] for training details.

## Evaluation Results

[![Coding benchmarks comparing Naive-N0.5-Flash with other models across seven software engineering and agentic tasks.][coding-figure]][coding-pdf]

<p align="center"><em>Figure 2. Coding and agentic task results. Naive-N0.5-Flash is highlighted in yellow.</em></p>

[![AI R&D benchmarks covering PostTrainBench, MLE-bench-30, PaperBench, SOL-ExecBench, NanoChat AutoResearch, and NanoGPT SpeedRun.][ai-rd-figure]][ai-rd-pdf]

<p align="center"><em>Figure 3. AI research and systems optimization results. Metric directions are indicated in the figure.</em></p>

<a id="evaluation-notes"></a>

<details>
<summary>Evaluation setup and metric notes</summary>

**Evaluation setup.** Unless otherwise noted, our evaluations of Naive-N0.5-Flash use Claude Code 2.1.207 with a 1M-token context window, temperature 1.0, and top-p 0.95. The harness exposes only basic file I/O and Bash tools.

**Sources for reported benchmark scores are as follows:**

- **GLM-5.3 and GLM-5.3-Flash:** [GLM-5.3 blog](https://z.ai/blog/glm-5.3) and [GLM-5.3-Flash blog](https://z.ai/blog/glm-5.3-flash), respectively.
- **Kimi-K3:** [Kimi-K3 model page](https://huggingface.co/moonshotai/Kimi-K3).
- **Qwen-3.8-Max:** [Qwen-3.8-Max blog](https://qwen.ai/blog?id=qwen3.8).
- **Hy4-preview:** [Hy4-preview model page](https://huggingface.co/tencent/Hy4-preview).
- **DeepSeek-V4.1-Flash:** [DeepSeek-V4.1-Flash model page](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
- **Step-5-preview:** [Step-5-Preview-BF16 model page](https://huggingface.co/TypeSafeAI/Step-5-Preview-BF16).
- **Fable-5 (w/ fallback):** [GLM-5.3 blog](https://z.ai/blog/glm-5.3).
- **SWE-Bench Pro:** GPT-5.6-Sol, Opus-5, and Opus-5.5 scores are drawn from the [GPT-5.6 blog](https://openai.com/index/gpt-5-6/) and the [Claude Opus 5.5 System Card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf).
- **DeepSWE v1.1:** The Muse-Spark-1.3 score comes from its [Muse-Spark-1.3 blog](https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology). GPT-5.6-Sol and Opus-5 scores come from the [DeepSWE v1.1 leaderboard](https://deepswe.datacurve.ai/). The Opus-5.5 score comes from the [Claude Opus 5.5 System Card](https://www-cdn.anthropic.com/fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card.pdf).
- **Terminal-Bench 2.1:** Muse-Spark-1.3, GPT-5.6-Sol, and Opus-5 scores come from the [Muse-Spark-1.3 blog](https://research.meta.ai/static/muse-spark-1-3-multimodal-evaluation-methodology). The GPT-6-Astra score comes from the [Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/?version=2.1).
- **ALE-CLI:** GPT-5.6-Sol, GPT-6-Astra, Muse-Spark-1.3, Opus-5, and Opus-5.5 scores come from the [ALE-CLI leaderboard](https://agents-last-exam.org/leaderboard).
- **FrontierSWE v1:** We calculate the Dominance score using the competing systems’ results as of August 23, 2026.
- **ProgramBench:** We report the Almost@1 score. GPT-5.6-Sol and Opus-5 scores come from the [ProgramBench leaderboard](https://programbench.com/).
- **MLE-bench-30:** Gemini-3.5-Flash, Gemini-3.6-Flash, Grok-4.5, and GPT-5.6-Luna scores come from the [Gemini 3.6 Flash model card](https://deepmind.google/models/model-cards/gemini-3-6-flash/). Following its evaluation protocol, we report the average position score of Naive-N0.5-Flash.
- **PaperBench:** MiniMax M3, Opus-4.7, GPT-5.5, and Gemini-3.1-Pro scores come from the [MiniMax M3 model page](https://huggingface.co/MiniMaxAI/MiniMax-M3).
- **SOL-ExecBench, NanoChat AutoResearch, and NanoGPT SpeedRun:** Naive-N0.5-Flash scores were obtained using our in-house AutoResearch harness. Recursive Superintelligence Inc. scores come from its [research article](https://www.recursive.com/articles/first-steps-toward-automated-ai-research). Following the SOL-ExecBench update, we use its updated score from the [leaderboard](https://research.nvidia.com/benchmarks/sol-execbench).

</details>

<a id="quickstart"></a>
<a id="deployment"></a>

## Deployment

Naive-N0.5-Flash supports FP8 mixed-precision inference. For general use, we recommend setting the sampling parameters to `temperature=1.0` and `top_p=0.95`.

### Quick Start with Transformers

Naive-N0.5-Flash requires FP8-capable NVIDIA GPUs. The model weights occupy approximately 315 GB; allow additional GPU memory for inference.

```bash
pip install "transformers[torch,kernels]>=5.17.0"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "NaiveAI/Naive-N0.5-Flash-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=2048)
response = tokenizer.decode(
    output[0, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print(response)
```

## License

Naive-N0.5-Flash is released under the MIT License.

## Citation

If you find Naive-N0.5-Flash useful in your research or work, please cite:

```bibtex
@misc{naiveai2026naiven05flash,
  title  = {Naive-N0.5-Flash: Building Frontier AI with AI},
  author = {{NaiveAI Team}},
  year   = {2026},
  url    = {https://naive.ai/en/research/}
}
```

## Acknowledgments

Naive-N0.5-Flash builds on the work of the open-source community and gives back to it. We thank the Xiaomi MiMo team for making their MiMo-V2.5 base model publicly available, the DeepSeek team for their work on DeepSeek Sparse Attention (DSA), and the SGLang team and community for their open-source inference infrastructure.

## Contact

For questions, feedback, or collaboration, please contact us at contact@naive.ai or follow us on X at @naiveailab. You can also find our open-source projects and model releases on [GitHub](https://github.com/naiveai-labs) and [Hugging Face](https://huggingface.co/NaiveAI).

[website]: https://naive.ai/
[blog]: https://naive.ai/en/research/
[github]: https://github.com/NaiveAI-Labs/Naive-N0.5-Flash
[weights]: https://huggingface.co/NaiveAI/Naive-N0.5-Flash
[discussions]: https://huggingface.co/NaiveAI/Naive-N0.5-Flash/discussions
[base-model]: https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Base

<!-- Figures use bundled assets for preview and HF upload. To share one image source with the blog, replace these three paths with public image URLs. -->
[architecture-figure]: assets/hybrid-swa-dsa-architecture.svg
[coding-figure]: assets/coding-benchmarks.png
[coding-pdf]: assets/coding-benchmarks.pdf
[ai-rd-figure]: assets/ai-rd-benchmarks.png
[ai-rd-pdf]: assets/ai-rd-benchmarks.pdf
