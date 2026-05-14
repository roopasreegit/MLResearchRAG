eval_data = [
    {
    "question": "How does Low-Rank Adaptation (LoRA) reduce the number of trainable parameters during fine-tuning?",
    "ground_truth": "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, drastically reducing the parameters that need to be updated."
  },
    {
  "question": "I am looking into how to make language models more reliable when they encounter tasks requiring factual precision. Should I focus on changing how the model processes information during its actual deployment, or is it better to modify the underlying structural parameters before it ever runs?",
  "ground_truth": "This query touches on the core tension between inference-time optimization and parameter-driven modifications. To address deployment-time processing, the researcher should look into Retrieval-Augmented Generation frameworks like Self-RAG or CRAG, which dynamically evaluate and fetch facts during inference, or 'LLM in a flash' for runtime memory management. To address structural parameter modifications, they should examine parameter-efficient fine-tuning (PEFT) methods like LoRA and QLoRA, which permanently alter a small subset of structural weights to adapt the model's behavior prior to deployment."
},
  {
  "question": "Both Self-RAG and RLHF use internal mechanisms to score or critique text. How do the roles of these scoring mechanisms fundamentally differ regarding when they are applied and what they alter?",
  "ground_truth": "The mechanisms differ in timing and purpose: In RLHF, a separate reward model is used strictly during the offline training phase to optimize the policy of the generator via reinforcement learning. In contrast, Self-RAG embeds the critique mechanism directly into the generator at inference time using special 'reflection tokens', allowing the model to dynamically decide whether to retrieve data or alter its generation paths on the fly."
  },
  {
    "question": "Which architecture claims to break the trade-off between the parallel training speeds of modern architectures and the low-cost, step-by-step token generation efficiency of traditional recurrent models?",
    "ground_truth": "The 'Retentive Network' (RetNet) paper. It introduces a retention mechanism that allows for parallel training (similar to Transformers) and recurrent O(1) inference deployment (similar to RNNs)."
  },
  {
    "question": "If my system fetches low-quality, incorrect, or completely irrelevant data from the database, which framework acts as a safety net by cross-checking the quality and pulling in external web knowledge to fix it?",
    "ground_truth": "The Corrective Retrieval-Augmented Generation (CRAG) paper. It uses a lightweight evaluator to assess retrieval quality and triggers an external web search to correct or supplement the context if the internal retrieval is inaccurate."
  },
  {
    "question": "How does Learning from Human Feedback (RLHF) optimize text summarization quality?",
    "ground_truth": "It trains a reward model based on human preferences between alternative summaries, and then uses Reinforcement Learning (specifically Proximal Policy Optimization) to fine-tune the language model to maximize the predicted reward."
  },
  {
    "question": "What core architecture does the 'Attention Is All You Need' paper introduce, and what mechanism does it rely on to replace recurrence?",
    "ground_truth": "The paper introduces the Transformer architecture, which relies entirely on self-attention mechanisms to model global dependencies between input and output, completely replacing recurrent neural networks (RNNs) and convolution."
  },
  {
    "question": "What does the abbreviation BERT stand for, and how is it pre-trained to understand bidirectional context?",
    "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers. It is pre-trained using a Masked Language Model (MLM) objective, where random tokens are masked and the model predicts them using both left and right context, as well as a Next Sentence Prediction (NSP) task."
  },
  {
    "question": "What is the core concept of Chain of Thought (CoT) prompting?",
    "ground_truth": "Chain of Thought prompting improves the reasoning capabilities of large language models by prompting them to generate a series of intermediate deliberative reasoning steps before producing the final answer to a complex problem."
  },
  {
    "question": "How does Knowledge Distillation work according to Geoffrey Hinton's paper 'Distilling the Knowledge in a Neural Network'?",
    "ground_truth": "Knowledge distillation transfers knowledge from a large, cumbersome teacher model (or an ensemble of models) to a smaller, more efficient student model by training the student to mimic the 'soft targets' (class probabilities) produced by the teacher."
  },
  {
    "question": "What original model and training paradigm does the paper 'Improving language understanding by generative pre-training' introduce?",
    "ground_truth": "The paper introduces the original GPT (GPT-1) model. It uses a two-stage training paradigm consisting of unsupervised generative pre-training on a large diverse corpus using a causal language modeling objective, followed by supervised fine-tuning on specific downstream tasks."
  },
  {
    "question": "What is the main technique used in 'LLM in a flash' to efficiently run large language models on devices with limited DRAM?",
    "ground_truth": "The paper stores the model parameters in non-volatile flash memory (which is larger but slower) and dynamically loads them into DRAM only when needed, minimizing data transfer using techniques like windowing and sparsity-aware activation."
  },

  {
    "question": "What is the 'Lost in the Middle' phenomenon in long-context language models?",
    "ground_truth": "The 'Lost in the Middle' phenomenon is the tendency of language models to achieve the highest performance when relevant information is located at the absolute beginning or the end of the input context, while performance significantly degrades when the relevant data is placed in the middle."
  },
  {
    "question": "What are the key architectural enhancements of Mistral 7B compared to baseline Transformers?",
    "ground_truth": "Mistral 7B utilizes Grouped-Query Attention (GQA) for faster inference and lower memory footprint, and Sliding Window Attention (SWA) to handle longer sequences effectively with reduced computational overhead."
  },
  {
    "question": "According to the 'RAG vs Fine-Tuning' analysis, what are the primary use-case differentiators between the two approaches?",
    "ground_truth": "RAG is ideal for accessing dynamic, frequently updated, or external facts without retraining costs, whereas fine-tuning is better suited for adjusting the model's style, format, tone, or deep domain-specific behavioral alignment."
  },
  {
    "question": "What paradigm does the 'Retentive Network' (RetNet) propose to challenge Transformers, and what are its dual deployment advantages?",
    "ground_truth": "RetNet introduces a retention mechanism that supports both parallel training (like Transformers) and recurrent representation for O(1) inference cost (like RNNs), effectively breaking the impossible triangle of training parallelism, low inference cost, and high performance."
  },
  {
    "question": "What are 'reflection tokens' in the Self-RAG framework?",
    "ground_truth": "Reflection tokens are special output tokens generated by the model to self-evaluate and critique its own generation quality, assessing whether retrieval is necessary, if the retrieved context is relevant, and if the final answer is supported by the context."
  },
  {
    "question": "What is the primary objective of Corrective Retrieval-Augmented Generation (CRAG)?",
    "ground_truth": "CRAG evaluates the relationship between the query and the retrieved documents using a lightweight retrieval evaluator. It then corrects or refines the context by incorporating web search if the retrieval is deemed inaccurate, or stripping irrelevant text if it is only partially correct."
  },
  {
    "question": "How does QLoRA further optimize the memory efficiency of baseline LoRA fine-tuning?",
    "ground_truth": "QLoRA quantizes the base model's weights to a high-precision 4-bit NormalFloat (NF4) data type, uses Double Quantization to reduce memory footprint from quantization constants, and implements Paged Optimizers to manage memory spikes during gradient updates."
  },
  {
    "question": "Which paper addresses the problem where an LLM suffers from 'amnesia' regarding facts buried deep inside the middle of a massive block of text?",
    "ground_truth": "The 'Lost in the Middle' paper addresses this, proving that language models are great at retrieving information at the very beginning or end of a context window, but struggle significantly when the relevant information is located in the middle."
  }
  
]