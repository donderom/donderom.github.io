---
title: "Hugging Face pipeline inference optimization"
description: "Impact of optimization techniques on Hugging Face Transformers pipeline performance"
date: 2023-02-19T14:43:36+02:00
draft: false
---

The goal of this post is to show how to apply a few practical optimizations to improve inference performance of [🤗 Transformers](https://huggingface.co/docs/transformers/index) pipelines on a single GPU. Compatibility with pipeline API is the driving factor behind the selection of approaches for inference optimization. This is a practical guide to optimizing inference of 🤗 Transformers pipelines based on my personal experience. For more methods on how to make transformer inference more efficient, I recommend checking out Lilian Weng's blog{{< sidenote >}}[Weng, Lilian. (Jan 2023). Large Transformer Model Inference Optimization. Lil’Log](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).{{< /sidenote >}}.

All the code is run on NVIDIA RTX 4090 24G using Python 3.10 and PyTorch 1.13.1.
Other dependencies are:
```
datasets==2.9.0
evaluate[evaluator]==0.4.0
optimum[onnxruntime-gpu]==1.6.4
transformers==4.26.1
```

To start off, let's establish the baseline and evaluation method, which will remain consistent across all approaches.

## Baseline

As a baseline, I'm going to use the [RoBERTa base](https://huggingface.co/roberta-base) model fine-tuned on the [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) dataset for extractive question answering. The training is done using the scripts from the [transformers](https://github.com/huggingface/transformers) [examples](https://github.com/huggingface/transformers/tree/v4.26.1/examples/pytorch/question-answering){{< sidenote >}}All training examples are based on the `v4.26.1` tag.{{< /sidenote >}} for PyTorch.

First, let's fine-tune the `roberta-base` model on the `squad` dataset for two epochs with the following parameters (to match the examples from the `transformers`):

```sh
python run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name rajpurkar/squad \
  --do_train \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --save_strategy no \
  --output_dir /workspace/roberta-base-squad
```

### Pipeline Evaluation

The following code is going to be used for the 🤗 Transformers pipeline{{< sidenote >}}The official tutorial on how to use pipelines for inference can be found [here](https://huggingface.co/docs/transformers/pipeline_tutorial).{{< /sidenote >}} evaluation:

```py
from datasets import load_dataset
from evaluate import evaluator
from transformers import pipeline

Metrics = dict[str, float]

batch_size = 8
data = load_dataset('rajpurkar/squad', split='validation')
qa_evaluator = evaluator('question-answering')
def qa_eval(model, tokenizer) -> Metrics:
    pipe = pipeline(
        task='question-answering',
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=0,
    )
    metrics = qa_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        metric='squad',
        device=0,
    )
    return {k: round(v, 4) for k, v in metrics.items()}
```

With the fine-tuned model and evaluation method in place, we can establish the baseline as follows:

```py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = '/workspace/roberta-base-squad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_eval(model=model, tokenizer=tokenizer)

{'exact_match': 85.6481, 'f1': 92.1859, 'samples_per_second': 293.3532}
```

The accuracy metrics are `exact_match` and `f1`, while the throughput is reported as `samples_per_second`. The important thing to note is that the numbers are for the pipeline and not the model itself, as the pipeline has extra logic for computing the best answer. Additionally, there is overhead caused by the evaluation. According to which, the pipeline baseline is indicated by an `f1` score of 92.1859 and a throughput of 293 samples per second. The exact numbers are not that significant as relative performance improvements compared to the baseline.

With the baseline and evaluation out of the way, let's see how to apply the first optimization technique---knowledge distillation.

## Knowledge Distillation

*Knowledge distillation* is a training technique for transferring knowledge from a pre-trained model ("teacher") to a less complex one ("student").

In some cases, task-specific knowledge distillation refers to the process of fine-tuning already distilled language model such as DistilBERT. Let's see what we'll get by fine-tuning the [DistilRoBERTa base](https://huggingface.co/distilroberta-base) model on the SQuAD with the same parameters:

```sh
python run_qa.py \
  --model_name_or_path distilroberta-base \
  --dataset_name rajpurkar/squad \
  --do_train \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --save_strategy no \
  --output_dir /workspace/distilroberta-base-squad
```

Evaluation:
```py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = '/workspace/distilroberta-base-squad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_eval(model=model, tokenizer=tokenizer)

{'exact_match': 80.5109, 'f1': 87.7095, 'samples_per_second': 381.8372}
```

Although a 5% drop in pipeline accuracy might be the biggest trade-off to make, a 30% increase in speed is still quite appealing, especially when combined with other techniques. Let's see if we can do better with two-step knowledge distillation.

### Two-step distillation

{{< blockquote cite="https://arxiv.org/abs/1910.01108" footer="Sanh, V.; Debut, L.; Chaumond, J. & Wolf, T. (2019), 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter'" >}}
We also studied whether we could add another step of distillation during the adaptation phase by fine-tuning DistilBERT on SQuAD using a BERT model previously fine-tuned on SQuAD as a teacher for an additional term in the loss (knowledge distillation). In this setting, there are thus two successive steps of distillation, one during the pre-training phase and one during the adaptation phase. In this case, we were able to reach interesting performances given the size of the model: 79.8 F1 and 70.4 EM, i.e. within 3 points of the full model.
{{< /blockquote >}}

Let me show how to implement two-step knowledge distillation to improve accuracy while keeping the throughput. In order to do this, we need to add one file to the same folder where the `run_qa.py` script is (`examples/pytorch/question-answering`):

`distil_trainer_qa.py`
```py
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, TrainingArguments

from trainer_qa import QuestionAnsweringTrainer


@dataclass
class DistilTrainingArguments(TrainingArguments):
    alpha: float = field(
        default=0.25,
        metadata={"help": "Controls the relative strength of each loss"}
    )

    temperature: int = field(
        default=3,
        metadata={"help": "Scaling factor to soften probabilities"}
    )

    teacher_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model name or path"}
    )


class DistilQuestionAnsweringTrainer(QuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = None
        if self.args.teacher_name_or_path:
            self.teacher = AutoModelForQuestionAnswering.from_pretrained(
                self.args.teacher_name_or_path
            )
            self.teacher.to(self.args.device).eval()
        self.loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        self.alpha = self.args.alpha
        self.T = self.args.temperature
        self.T2 = self.T ** 2

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher:
            outputs_stu = model(**inputs)
            with torch.no_grad():
                outputs_tea = self.teacher(**inputs)
            loss_start = self.T2 * self.loss_fn(
                F.log_softmax(outputs_stu.start_logits / self.T, dim=-1),
                F.softmax(outputs_tea.start_logits / self.T, dim=-1)
            )
            loss_end = self.T2 * self.loss_fn(
                F.log_softmax(outputs_stu.end_logits / self.T, dim=-1),
                F.softmax(outputs_tea.end_logits / self.T, dim=-1)
            )
            loss_kd = (loss_start + loss_end) / 2.0
            # Overall loss as a weighted sum
            loss = self.alpha * outputs_stu.loss + (1.0 - self.alpha) * loss_kd
            return (loss, outputs_stu) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs)
```

It extends the `TrainingArguments` by adding three new parameters:
* `alpha`: Controls how much weight is put on the student-teacher loss relative to the student loss alone, with higher value giving more weight to the student loss.
* `temperature`: Controls the level of smoothing of the teacher probability distribution, with higher temperature leading to a softer distribution.
* `teacher_name_or_path`: Name or path to the pre-trained teacher model.

The default values for `alpha` and `temperature` are set as a result of a dozen of [Optuna](https://github.com/optuna/optuna) trials{{< sidenote >}}Hyperparameter space: 𝛼 ∈ [0, 1], T ∈ [1, 10].{{< /sidenote >}} for this specific task. In general, for many NLP tasks a good starting point would be `alpha < 0.3` and `temperature > 1`.

The `DistilQuestionAnsweringTrainer` class extends the `QuestionAnsweringTrainer` from the `trainer_qa.py` file by overriding the `compute_loss` function. The new loss function involves logits from the teacher to calculate the overall loss as a weighted sum of the student's loss and the distillation loss using the new `alpha` and `temperature` parameters.

In addition to this new file, the `run_qa.py` itself should be modified to support the new distillation trainer:

`run_qa.py`
```diff
  from utils_qa import postprocess_qa_predictions
+ from distil_trainer_qa import DistilTrainingArguments, DistilQuestionAnsweringTrainer

- parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)
+ parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistilTrainingArguments))

- trainer = QuestionAnsweringTrainer(
+ trainer = DistilQuestionAnsweringTrainer(
```

If the changes have been applied correctly, training with two-step knowledge distillation should be as easy as adding the `--teacher` parameter (when leaving the `alpha` and the `temperature` parameters as default):

```sh {hl_lines=10}
python run_qa.py \
  --model_name_or_path distilroberta-base \
  --dataset_name rajpurkar/squad \
  --do_train \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --save_strategy no \
  --output_dir /workspace/distilroberta-base-distilsquad \
  --teacher /workspace/roberta-base-squad
```

If there is any justice in the world, bringing teacher into the equation should move accuracy closer to our baseline.

```py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = '/workspace/distilroberta-base-distilsquad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_eval(model=model, tokenizer=tokenizer)

{'exact_match': 81.6746, 'f1': 88.7431, 'samples_per_second': 384.2462}
```

The two-step distillation process maintains a throughput of 31% over the baseline, retaining 96% of the baseline accuracy. Increasing the number of epochs (and playing with other hyperparameters) will bump this number even further{{< sidenote >}}The [DistilBERT](https://arxiv.org/abs/1910.01108) paper reports "within 3 points of the full model".{{< /sidenote >}} without any negative effect on throughput. I will use this distilled model for the upcoming set of methods.

## Automatic Mixed Precision

*Automatic Mixed Precision (AMP)* allows to use a mix of `torch.float32` and half-precision (`torch.float16`) floating point datatypes during inference, thereby reducing the memory footprint and improving performance while maintaining accuracy.

```py
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = '/workspace/distilroberta-base-distilsquad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# autocast context manager allows code regions to run in mixed precision
with torch.cuda.amp.autocast(dtype=torch.float16):
    print(qa_eval(model=model, tokenizer=tokenizer))

{'exact_match': 81.7029, 'f1': 88.7576, 'samples_per_second': 416.5399}
```

In addition to the performance improvement by 8.4% compared to the original model{{< sidenote >}}Double that for the non-distilled baseline model.{{< /sidenote >}}, there is also a nearly 0.03% increase in accuracy as well. In general, it's a safe approach to use the AMP `autocast` with language models. Later on, we'll see that, even in situations where it does not impact the performance, it typically does not cause any harm.

## TorchScript

*TorchScript* is a way to create serializable and optimizable models from PyTorch code{{< sidenote >}}From PyTorch documentation on [TorchScript](https://pytorch.org/docs/stable/jit.html).{{< /sidenote >}}.

There are two ways to use TorchScript models with pipelines:

* Tracing an existing model.
* Loading a TorchScript module to use for inference.

We'll now explore how to trace an existing model to use for pipeline inference, as loading a TorchScript module will be straightforward with tricks used for tracing.

There are three steps to trace a model to use it with pipeline:

1. Prepare example inputs that will be passed to the model.
2. Trace the model to capture its structure by evaluating it using example inputs.
3. Patch existing model to use TorchScript for forward pass.

```py {hl_lines=38}
from datasets import load_dataset
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from transformers.modeling_outputs import QuestionAnsweringModelOutput

model_name = '/workspace/distilroberta-base-distilsquad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name, torchscript=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(examples):
    return tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=384,
        stride=128,
        padding='max_length',
    )

# 1. Prepare example inputs
squad = load_dataset('rajpurkar/squad', split=f'validation[:{batch_size}]')
example_batch = squad.map(
    encode,
    batched=True,
    remove_columns=squad.column_names
).with_format('torch')

# 2. Trace the model
jit_inputs = (
    example_batch['input_ids'].to('cuda'),
    example_batch['attention_mask'].to('cuda')
)
jit_model = torch.jit.trace(model, jit_inputs)
jit_model = torch.jit.freeze(jit_model).to('cuda').eval()

# 3. Patch the model forward function to be compatible with pipeline
def forward_qa_wrap(**inputs):
    with torch.jit.optimized_execution(False):
        start_logits, end_logits = jit_model(**inputs)
    return QuestionAnsweringModelOutput(
        start_logits=start_logits,
        end_logits=end_logits
    )

model.forward = forward_qa_wrap

with torch.cuda.amp.autocast(dtype=torch.float16):
    print(qa_eval(model=model, tokenizer=tokenizer))

{'exact_match': 81.7029, 'f1': 88.7576, 'samples_per_second': 446.3015}
```

The performance increase is 16% compared to the original distilled model, which is almost twice the improvement achieved by `autocast` alone.

There are several important tricks to get there:

* TorchScript is optimized using just-in-time (JIT) compilation based on inputs. When inputs are different (and they are in the case of question answering), JIT recompilation will slow down the pipeline performance eventually. To keep the performance stable, it's essential to run TorchScript inside the `torch.jit.optimized_execution(False)` context manager.

* The pipeline for question answering expects a model to return an instance of the `QuestionAnsweringModelOutput` class, while TorchScript will return a tuple of logits. To make it compatible with the pipeline API, it should be wrapped in a function mimicking the forward call. The original model's forward function is then replaced with the new wrapper{{< sidenote >}}This is a more general technique that could be used to wrap the NVIDIA [Apex](https://github.com/nvidia/apex) for example.{{< /sidenote >}}.

* Using TorchScript along with `autocast` is crucial for inference optimization.  In my experience, utilizing `autocast` during inference instead of tracing itself results in a slight improvement in performance.

While this approach may result in less noticeable performance improvements on some GPUs, such as the RTX 30 series, it is more stable from run to run than AMP alone.
  
## DeepSpeed

Setting up [DeepSpeed](https://www.deepspeed.ai) integration requires more effort if there are missing system dependencies, such as Python shared libs and header files{{< sidenote >}}More about DeepSpeed installation [here](https://www.deepspeed.ai/tutorials/advanced-install).{{< /sidenote >}}. If you are using DeepSpeed for training, you can leverage [DeepSpeed-Inference](https://www.deepspeed.ai/inference) to perform inference as well. Despite the fact that DeepSpeed-Inference shares its name with DeepSpeed, it does not use [ZeRO](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer) technology.

Using DeepSpeed-Inference with the pipeline API is straightforward{{< sidenote >}}DeepSpeed-Inference [tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial).{{< /sidenote >}}:
```py
import deepspeed
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = '/workspace/distilroberta-base-distilsquad'
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

engine = deepspeed.init_inference(
    model,
    dtype=torch.float16,
    replace_with_kernel_inject=True,
)

with torch.cuda.amp.autocast(dtype=torch.float16):
    print(qa_eval(model=engine.module, tokenizer=tokenizer))
```

Although I don't have the exact numbers for the same GPU used in the other methods, it consistently outperforms the previous approach (TorchScript with AMP) by 5--7% on the RTX 3090 24G.
  
## Optimum

[🤗 Optimum](https://github.com/huggingface/optimum) is an extension of 🤗 Transformers, providing a set of  performance optimization tools with unified API to train and run models on targeted hardware with maximum efficiency{{< sidenote >}}More about 🤗 Optimum can be found in the [official documentation](https://huggingface.co/docs/optimum/index).{{< /sidenote >}}. Optimum can be used for accelerated inference with built-in support for transformers pipelines, making it an ideal candidate for pipeline optimization techniques. I will demonstrate how to apply graph optimization to accelerate inference with ONNX Runtime.

[ONNX Runtime](https://onnxruntime.ai) (ORT) is a cross-platform, high-performance engine for [Open Neural Network Exchange (ONNX)](https://onnx.ai) models used to accelerate inference and training of machine learning models.

The code below outlines every step required for 🤗 Optimum inference using ONNX Runtime with relevant comments:

```py
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoTokenizer

model_name = '/workspace/distilroberta-base-distilsquad'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a model and export it to ONNX format
model = ORTModelForQuestionAnswering.from_pretrained(model_name, from_transformers=True)

# Define optimization configuration
opt_config = OptimizationConfig(
    optimization_level=2,
    optimize_for_gpu=True,
    fp16=True,
)

# Create optimizer
optimizer = ORTOptimizer.from_pretrained(model)

# Optimize the model applying defined optimization strategy
optimizer.optimize(
    optimization_config=opt_config,
    save_dir=f'{model_name}/onnx',
    file_suffix=None,
)

# Load the optimized model for inference
model = ORTModelForQuestionAnswering.from_pretrained(model_name, file_name='onnx/model.onnx')

qa_eval(model=model, tokenizer=tokenizer)

{'exact_match': 81.7029, 'f1': 88.7575, 'samples_per_second': 477.7468}
```

The throughput of the optimized model improved by 24% compared to the initial distilled model.

A few notes on the example configuration.

Since our focus is on GPU inference, setting the `optimize_for_gpu` to `True` is crucial. There is no specific reason to go with the `optimization_level=2` other than it being a good starting point. If mixed precision is enabled (`fp16=True`), there is no need to use `autocast` during inference. However, if the optimization configuration being used is unknown, it's ok to use both `autocast` and optimized models together.

## Numerology

Ok, these are some of the optimization techniques that work well with 🤗 Transformers pipelines in my experience. As most of the examples are based on the distilled model, let's see the results with the same techniques being applied to the baseline model too. The table can be mentally broken down into two parts---the first part without distillation, and the second part containing optimizations combined with distillation.

| Approach                         | Exact Match | F1        | F1 diff | Samples/sec | Speedup |
| :---                             | ---:        | ---:      | ---:    | ---:        | ---:    |
| Baseline                         | 85.65       | 92.19     | 100%    | 293.4       | 1.00    |
| Automatic Mixed Precision (AMP)  | 85.64       | 92.19     | 100%    | 347.2       | 1.27    |
| TorchScript + AMP                | 85.64       | 92.19     | 100%    | 379.8       | 1.29    |
| Optimum (ORT)                    | **85.7**    | **92.21** | 100%    | 429.6       | 1.46    |
| Knowledge Distillation           | 81.67       | 88.74     | 96%     | 384.2       | 1.31    |
| Distillation + AMP               | 81.7        | 88.76     | 96%     | 416.5       | 1.42    |
| Distillation + TorchScript + AMP | 81.7        | 88.76     | 96%     | 446.3       | 1.52    |
| Distillation + Optimum (ORT)     | 81.7        | 88.76     | 96%     | **477.7**   | 1.63    |

Although the difference is negligible, using 🤗 Optimum with ONNX Runtime results in slightly higher accuracy compared to the baseline model. What's even better is that all the described methods have no negative impact on accuracy, except for knowledge distillation. It looks like that, in order to surpass the 1.5x throughput speedup, one may have to sacrifice some accuracy. While this threshold is specific to the particular hardware, the general trend should be similar for different GPUs. So it's not surprising that the highest throughput is achieved by using 🤗 Optimum in combination with knowledge distillation, as these are the two most significant contributing factors.
