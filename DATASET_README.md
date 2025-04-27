---
language:
- en
- sto # Stoney Nakoda
license: apache-2.0 # Or another license if you prefer - common choices are apache-2.0, mit, cc-by-sa-4.0
tags:
- translation
- question-answering
- low-resource
- synthetic-data
- stoney-nakoda
- indigenous-language
datasets:
- HarleyCooper/synthetic_stoney_data
pipeline_tag: text2text-generation
pretty_name: "Synthetic Stoney Nakoda Q&A Dataset"
---

# Dataset Card for Synthetic Stoney Nakoda Q&A

## Dataset Description

This dataset contains 150,000 synthetic question-answer pairs designed for training language models in Stoney Nakoda and English. It was generated as a foundational resource to aid in the development of NLP tools for the low-resource Stoney Nakoda language. The pairs cover translations, grammatical nuances, contextual usage, and cultural relevance derived from bilingual dictionary entries.

**Homepage:** [StoneyNakoda GitHub Repository](https://github.com/HarleyCoops/StoneyNakoda.git)
**Repository:** [StoneyNakoda GitHub Repository](https://github.com/HarleyCoops/StoneyNakoda.git)
**Point of Contact:** Harley Cooper

### Fingerprinting Note

I intentionally trained this synthetic data with a meticulous and total exclusion of all words starting with a random Stoney Nakoda letter. This was done to create a fingerprint for this synthetic dataset so future researchers can identify and exclude if needed.

Only myself and any curious native speaker will notice the difference. There is no impact on model performance.

## Languages

The dataset contains text in English (`en`) and Stoney Nakoda (`sto`).

## Dataset Structure

The dataset is provided in JSON Lines (`.jsonl`) format. Each line represents a single question-answer pair.

### Data Fields

Each JSON object has the following fields:

*   `question`: (string) The question, which can be in English or request information/translation related to Stoney Nakoda.
*   `answer`: (string) The corresponding answer, providing the translation, explanation, or context.
*   `source_language`: (string) Indicates whether the generation prompt focused on the 'english' or 'stoney' perspective of the dictionary entry (`english` or `stoney`).
*   `generated_at`: (string) ISO 8601 timestamp indicating when the pair was generated.
*   `pair_id`: (integer) A unique identifier for the Q&A pair.
*   `method`: (string, optional) Indicates the generation method (e.g., 'gemini', 'synonym'). *Note: This field appears in the sample data but wasn't explicitly mentioned in the generation script description.*
*   `original_id`: (integer, optional) An ID potentially linking back to the source dictionary entry. *Note: This field appears in the sample data but wasn't explicitly mentioned in the generation script description.*

### Data Splits

The dataset consists of a single split: `train`, containing all 150,000 Q&A pairs.

### Example Data Point

```json
{
    "question": "How do the Stoney Nakoda terms 'wîja bare' and 'wîja eîchihnâgach' differ in meaning, and how do these distinctions reflect the concepts of wholeness and total commitment?",
    "answer": "The Stoney Nakoda term 'Wîja bare' signifies a complete sum or total amount. In contrast, 'wîja eîchihnâgach' describes a complete and unwavering commitment, a full and focused dedication to a particular endeavor. Both phrases convey the idea of wholeness, but one refers to a quantitative totality, while the other emphasizes a qualitative, personal devotion.",
    "generated_at": "2024-12-22T11:50:47.088545",
    "method": "gemini",
    "original_id": 12923,
    "source_language": "english", // Added based on likely generation context
    "pair_id": 1 // Added based on likely generation context
}
```

## Dataset Creation

### Curation Rationale

This dataset was generated synthetically as a first step towards creating robust NLP tools for Stoney Nakoda, a language with limited digital resources. While the ideal scenario involves community-created and validated data, this dataset serves as a foundational resource and proof-of-concept, reverse-engineered from existing bilingual dictionaries (`english_dictionary.jsonl`, `stoney_dictionary.jsonl`). It aims to provide a substantial volume of training data to bootstrap model development.

### Source Data

The source data comprises two JSONL files containing bilingual dictionary entries:
1.  `english_dictionary.jsonl`: English-to-Stoney Nakoda entries.
2.  `stoney_dictionary.jsonl`: Stoney Nakoda-to-English entries.

### Generation Process

The Q&A pairs were generated using a Python script (`bilingual_qa_generator.py`) leveraging Google's Gemini API (`gemini-2.0-exp`). The process involved:

1.  Reading entries from the source dictionary files.
2.  Grouping entries into small contexts (typically 5 entries).
3.  Creating detailed prompts instructing the Gemini model to act as a Stoney Nakoda language expert.
4.  The prompts guided the model to generate diverse Q&A pairs based on the provided dictionary context, focusing on:
    *   Bidirectional translation.
    *   Nuances between related terms.
    *   Grammatical classifications and usage.
    *   Contextual scenarios.
    *   Word relationships and patterns.
    *   Cultural context where applicable.
5.  The script generated 75,000 pairs from the English-to-Stoney perspective and 75,000 pairs from the Stoney-to-English perspective, totaling 150,000 pairs.
6.  Generated pairs were formatted as JSON objects and saved to the output `.jsonl` file.

### Annotations

The dataset does not contain human annotations. The Q&A pairs were generated by the Gemini language model based on the prompts and dictionary data described above.

### Personal and Sensitive Information

The source dictionaries and the generated data primarily contain linguistic information (words, translations, definitions, examples). No personally identifiable information (PII) is expected to be included.

## Considerations for Using the Data

### Social Impact

The primary goal of this dataset is positive: to support the revitalization and preservation of the Stoney Nakoda language by enabling the creation of language technologies. It provides a resource where few previously existed. However, users should be mindful that the language reflects a specific culture and community, and its use in technology should be respectful and ideally involve community consultation.

### Discussion of Biases

*   **Synthetic Data Bias:** As the data is AI-generated, it may reflect biases present in the underlying Gemini model or limitations in its understanding of Stoney Nakoda based solely on the provided dictionary entries. It might not capture the full spectrum of natural language use or cultural nuance.
*   **Source Dictionary Bias:** The original dictionaries may have their own biases or limitations in terms of coverage, dialect representation, or orthography.
*   **Fingerprinting:** The intentional exclusion of words starting with a specific letter, while useful for tracking, is an artificial manipulation of the data.

### Other Known Limitations

*   The data is synthetic and may not perfectly mirror real-world Stoney Nakoda usage.
*   The quality of the generated pairs depends on the quality of the source dictionaries and the capabilities of the generative model at the time of creation.
*   Validation by fluent Stoney Nakoda speakers has not been systematically performed on this version of the dataset.


### More Information

For more context on the project aiming to build language tools for Stoney Nakoda, please visit the main repository: [https://github.com/HarleyCoops/StoneyNakoda.git](https://github.com/HarleyCoops/StoneyNakoda.git)

## Intended Use

This dataset was created specifically for research into **mechanistic interpretability** and **reinforcement learning (RL)** applied to low-resource language translation, particularly Stoney Nakoda. The primary goal is to explore how language models reason through linguistic tasks and how this differs from reasoning in domains like mathematics or coding.

The envisioned training pipeline leverages this dataset in three stages:

1.  **Supervised Fine-Tuning (SFT) with LoRA:** Adapting a small, reasoning-capable base model (e.g., DeepSeek-R1-Zero-7B, Phi-3-Mini-3.8B) to the Stoney Nakoda lexicon and the Q&A format using Low-Rank Adaptation.
2.  **Reward Model Training:** Developing a model that evaluates not just the final translation quality but also the reasoning *trace* (Chain-of-Thought) leading to it. This involves a composite reward signal.
3.  **Policy Optimization with GRPO:** Using Group-Relative Policy Optimization (GRPO), an actor-critic-free RL algorithm suitable for smaller datasets, to fine-tune the model to maximize the composite reward, thereby improving both translation accuracy and reasoning validity.

### Composite Reward Design

A key aspect of the intended use is the development and application of a composite reward function $ R $ during the RL phase. This function aims to capture multiple facets of translation quality and reasoning:

$$
R = \lambda_1 R_{lex} + \lambda_2 R_{sem} + \lambda_3 R_{cot} + \lambda_4 R_{dict} + \lambda_5 R_{morph} - \lambda_6 P_{hall}
$$

Where the components represent:
*   R<sub>lex</sub>: Lexical match (e.g., scaled chrF++).
*   R<sub>sem</sub>: Semantic faithfulness (e.g., normalized COMET-Kiwi).
*   R<sub>cot</sub>: Reasoning-trace validity (evaluating groundedness, coherence, etc.).
*   R<sub>dict</sub>: Dictionary anchoring (ensuring use of attested terms).
*   R<sub>morph</sub>: Morphological integrity (checking diacritics, suffixes).
*   P<sub>hall</sub>: Hallucination penalty (e.g., negative Mauve divergence).

### Bridging from Math/Coding RL

This approach explicitly borrows and adapts concepts from RL research in mathematical reasoning and code generation, mapping them to the domain of low-resource translation:

| RL Idea (Math/Coding) | Translation Analogue         | Reward Term                 |
|-----------------------|------------------------------|-----------------------------|
| Unit-test pass rate   | Dictionary-lemma match       | R<sub>dict</sub>         |
| Code-coverage bonus   | Morphological-feature coverage| R<sub>morph</sub>       |
| Execution trace reward| Chain-of-thought validity    | R<sub>cot</sub>           |
| Brevity/latency penalty| Concise trace bonus          | R<sub>brev</sub> (Optional)|
| Functional correctness| Semantic faithfulness        | R<sub>sem</sub>           |

By using this dataset with the described methodology, researchers can investigate how models build and represent linguistic understanding in a low-resource setting and leverage insights from other reasoning domains to improve translation quality and interpretability. The goal is to foster models whose reward function explicitly values *how* they reason in Stoney as much as *what* they say.

## Additional Information

### Dataset Curators

*   Harley Cooper

### Licensing Information

The dataset is licensed under the [Apache License 2.0](LICENSE).

### Citation Information

If you use this dataset in your research, please cite it as follows:

```bibtex
@dataset{harley_cooper_2024_synthetic_stoney_data,
  author       = {Harley Cooper},
  title        = {Synthetic Stoney Nakoda Q\&A Dataset},
  month        = apr,
  year         = 2024,
  publisher    = {Hugging Face},
  version      = {1.0.0},
  doi          = {10.57967/hf/datasets/synthetic-stoney-data}, # Example DOI - Replace if you get one
  url          = {https://huggingface.co/datasets/HarleyCooper/synthetic_stoney_data}
}
```


For detailed implementation steps and the full research plan, please refer to the main project repository: [https://github.com/HarleyCoops/StoneyNakoda.git](https://github.com/HarleyCoops/StoneyNakoda.git) or associated documentation like `Instructions.txt`. 