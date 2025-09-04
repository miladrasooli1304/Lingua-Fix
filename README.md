# 🚀 Fine-Tuning Gemma-2B for Grammar Correction & Translation

This project demonstrates how to **fine-tune Google's Gemma-2B-Instruct model** using **LoRA adapters** with [Unsloth](https://github.com/unslothai/unsloth) and Hugging Face tools for two main tasks:  
1. **Grammar Correction (English)**  
2. **Machine Translation (English ➝ Persian)**  

---

## 📌 Project Overview  
Large Language Models (LLMs) like Gemma-2B are powerful but often require fine-tuning for specialized tasks. This project fine-tunes the **Gemma-2B-it** model to:  
- Correct grammatical errors in English sentences.  
- Translate English text to Persian.  

The fine-tuning process uses **LoRA** (Low-Rank Adaptation) for parameter-efficient training, leveraging the **Unsloth** library to optimize for low VRAM usage. The datasets used are **JFLEG** for grammar correction and **Opus100 (en-fa)** for English-to-Persian translation.

### Example
**Input**: `Correct the grammar of this sentence: she am doctorr.`  
**Output**: `She is a doctor.`

---

## 🛠️ Tech Stack & Libraries  
- [Python 3.8+](https://www.python.org/)  
- [Transformers](https://huggingface.co/docs/transformers/index)  
- [TRL](https://huggingface.co/docs/trl/index)  
- [PEFT](https://huggingface.co/docs/peft/index)  
- [Unsloth](https://github.com/unslothai/unsloth)  
- [Datasets](https://huggingface.co/docs/datasets/index)  
- [PyTorch](https://pytorch.org/)  

---

## 📂 Repository Structure  
```
📁 Fine-Tune-Gemma2
│── 📜 fine-tune_model_gemma2.ipynb      # Training code for fine-tuning
│── 📜 test_model_fine-tune.ipynb        # Testing the fine-tuned model
│── 📂 dataset                          # Datasets (JFLEG, Opus100)
│── 📂 Model_w_save                     # Saved LoRA adapters & tokenizer
│── 📜 README.md                        # Project documentation
```

---

## ⚙️ Installation & Requirements  
This project was developed on **Google Colab**. To run locally, install the required dependencies:

```bash
pip install torch transformers datasets peft trl unsloth
```

**Note**: For 4-bit quantization, you may need `bitsandbytes`. If you face issues on Windows, set `load_in_4bit=False` in the code.

---

## ⚡ Training Pipeline  
1. **Load Base Model**: `google/gemma-2b-it` from Hugging Face.  
2. **Apply LoRA Adapters**: Configure LoRA with `r=16`, `lora_alpha=32`, `lora_dropout=0.05` on `q_proj` and `v_proj` modules.  
3. **Prepare Datasets**:  
   - **JFLEG**: Grammar correction dataset with incorrect and corrected sentence pairs.  
   - **Opus100 (en-fa)**: English-to-Persian translation dataset (subset of 5,000 samples).  
   - Combine datasets and split into 90% training and 10% evaluation sets.  
4. **Train with SFTTrainer**: Use `TRL`’s `SFTTrainer` with a custom formatting function and training arguments (3 epochs, learning rate `2e-4`, AdamW optimizer).  
5. **Save Model & Tokenizer**: Store in `Model_w_save/` directory.

---

## ▶️ How to Run  

### 1. Training  
Run the `fine-tune_model_gemma2.ipynb` notebook to:  
- Load and preprocess datasets.  
- Fine-tune the model using LoRA and Unsloth.  
- Save the fine-tuned model and tokenizer to `Model_w_save/`.  

Key training parameters:  
- Batch size: 4 (with gradient accumulation steps=4)  
- Learning rate: `2e-4`  
- Epochs: 3  
- Optimizer: AdamW  

### 2. Testing  
Run the `test_model_fine-tune.ipynb` notebook to test the fine-tuned model.  

**Example Code**:
```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base model
base_model_name = "google/gemma-2b-it"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Load fine-tuned weights
lora_path = "./Model_w_save"
model = PeftModel.from_pretrained(model, lora_path)
FastLanguageModel.for_inference(model)

# Run inference
prompt = "Correct the grammar of this sentence:\nshe am doctorr."
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Output**:  
```
She is a doctor.
```

### 3. Streaming Output  
The test notebook also supports **token-by-token streaming** using `TextIteratorStreamer` for real-time output display.

---

## 📊 Training Results  
Below is a snippet of the training log:  

| Step | Loss   |
|------|--------|
| 56   | 1.6197 |
| 57   | 1.9192 |
| 58   | 1.9081 |
| 59   | 2.3043 |
| 60   | 2.3088 |

---

## 🌍 Applications  
- **Grammar Correction**: Improve the quality of English text for writing tools, educational platforms, or content editing.  
- **Translation**: Translate English to Persian for cross-lingual communication or localization.  
- **Extensibility**: The model can be adapted for other NLP tasks with additional datasets.  

---

## 👨‍💻 Author  
**Milad Rasooli**  
- 📧 **Email**: miladrasooli1304@gmail.com  
- 🌐 **GitHub**: [miladrasooli1304](https://github.com/miladrasooli1304)  

---

## ⭐ Acknowledgements  
- [Google Gemma](https://huggingface.co/google/gemma-2b-it)  
- [Unsloth](https://github.com/unslothai/unsloth)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  
- [JFLEG Grammar Dataset](https://github.com/keisks/jfleg)  
- [Opus100 Translation Dataset](https://huggingface.co/datasets/opus100)  

---

## 📜 License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.