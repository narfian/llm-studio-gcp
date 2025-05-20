# llm-studio-gcp

# Setup instructions to use Generative AI on Google Cloud

This folder contains instructions on:

- Setting up your Google Cloud project
- Notebook environments
  - Setting up Colab
  - Setting up Vertex AI Workbench
- Python SDK for Vertex AI

## Setting up your Google Cloud project

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager).
   When you first create an account, you get a $300 free credit towards your compute/storage costs.

2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

3. [Enable the Vertex AI API and Google Cloud Storage API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,storage.googleapis.com).

## Notebook environments

### Colab

[Google Colab](https://colab.research.google.com/) allows you to write and execute Python in your browser with minimal setup.

To use Colab with this repo, please click on the "Open in Colab" link at the top of any notebook file in this repo to launch it in Colab. Then follow the instructions within.

For Colab you will need to authenticate so that you can use Google Cloud from Colab:

```py
from google.colab import auth
auth.authenticate_user()
```

When using the vertexai Python SDK, you will also need to initialize it with your Google Cloud `project_id` and `location`:

```py
PROJECT_ID = "your-project-id"
LOCATION = "" #e.g. us-central1

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Vertex AI Workbench

[Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench) is the JupyterLab notebook environment on Google Cloud, which enables you to create and customize notebook instances. You do not need extra authentication steps.

#### Creating your notebook instance on Vertex AI Workbench

To create a new JupyterLab instance on Vertex AI Workbench, follow the [instructions here to create a user-managed notebooks instance](https://cloud.google.com/vertex-ai/docs/workbench/user-managed/create-new).

#### Using this repository on Vertex AI Workbench

After launching the notebook instance, you can clone this repository in your JupyterLab environment. To do so, open a Terminal in JupyterLab. Then run the command below to clone the repository into your instance:

```sh
git clone https://github.com/GoogleCloudPlatform/generative-ai.git
```

#### Local development

- Install the [Google Cloud SDK](https://cloud.google.com/sdk).

- Obtain authentication credentials. Create local credentials by running the following command and following the oauth2 flow (read more about the command [here](https://cloud.google.com/sdk/gcloud/reference/beta/auth/application-default/login)):

  ```bash
  gcloud auth application-default login
  ```

## Python library

Install the latest Python SDK:

```sh
%pip install google-cloud-aiplatform --upgrade
```

You will need to initialize `vertexai` with your `project_id` and `location`:

```py
PROJECT_ID = "your-project-id"
LOCATION = "" #e.g. us-central1

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)
```



## Data Handling Notebooks
- [Synthetic Data Gen](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-generation/synthetic_data_generation_using_gemini.ipynb)

- [Data Augmentation for Text](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb)

- [Document Parsing with llamaindex and Llamaparse](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/doc_parsing_with_llamaindex_and_llamaparse.ipynb)

- [Document Processing](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/document_processing.ipynb)

- [Summarization Large Documents](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/summarization_large_documents_langchain.ipynb)

## Finetuning Notebooks
- [VertexAI TRL Finetuning Gemma](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/fine-tuning/vertex_ai_trl_fine_tuning_gemma.ipynb)


## Model Garden SDK
- [Get Started with model Garden SDK](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/get_started_with_model_garden_sdk.ipynb)
> - Find the models that you can deploy
> - Deploy your 1st Model Garden model
> - Handle with some advanced usage including setting deployment parameters and error handling
- [Model Garden Axolotl Finetuning](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_axolotl_finetuning.ipynb)
- [Model Garden Deployment Tutorial](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_deployment_tutorial.ipynb)
- [Model Garden Finetuning Tutorial](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_finetuning_tutorial.ipynb)
- [Model Garden Gradio Streaming Chat Completion](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gradio_streaming_chat_completions.ipynb)
- []



## SmolAgent 
- [VertexAI Deepseek Smolagents](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/use-cases/vertex_ai_deepseek_smolagents.ipynb)



# Open Models

This repository contains examples for deploying and fine-tuning open source models with Vertex AI.
[Open Models URL](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/open-models)

### Serving

- [serving/cloud_run_ollama_gemma3_inference.ipynb](./serving/cloud_run_ollama_gemma3_inference.ipynb) - This notebook showcase how to deploy Google Gemma 3 in Cloud Run using Ollama, with the objective to build a simple API for chat.
- [serving/cloud_run_vllm_gemma3_inference.ipynb](./serving/cloud_run_vllm_gemma3_inference.ipynb) - This notebook showcase how to deploy Google Gemma 3 in Cloud Run using vLLM, with the objective to build a simple API for chat.
- [serving/vertex_ai_ollama_gemma2_rag_agent.ipynb](./serving/vertex_ai_ollama_gemma2_rag_agent.ipynb) - This notebooks provides steps and code to deploy an open source agentic RAG pipeline to Vertex AI Prediction using Ollama and a Gemma 2 model adapter.
- [serving/vertex_ai_pytorch_inference_paligemma_with_custom_handler.ipynb](./serving/vertex_ai_pytorch_inference_paligemma_with_custom_handler.ipynb) - This notebooks provides steps and code to deploy Google PaliGemma with the Hugging Face Python Inference DLC using a custom handler on Vertex AI.
- [serving/vertex_ai_pytorch_inference_pllum_with_custom_handler.ipynb](./serving/vertex_ai_pytorch_inference_pllum_with_custom_handler.ipynb) - This notebook shows how to deploy Polish Large Language Model (PLLuM) from the Hugging Face Hub on Vertex AI using the Hugging Face Deep Learning Container (DLC) for Pytorch Inference in combination with a custom handler.
- [serving/vertex_ai_text_generation_inference_gemma.ipynb](./serving/vertex_ai_text_generation_inference_gemma.ipynb) - This notebooks provides steps and code to deploy Google Gemma with the Hugging Face DLC for Text Generation Inference (TGI) on Vertex AI.
- [serving/vertex_ai_tgi_gemma_multi_lora_adapters_deployment.ipynb](./serving/vertex_ai_tgi_gemma_multi_lora_adapters_deployment.ipynb) - This notebook showcases how to deploy Gemma 2 from the Hugging Face Hub with multiple LoRA adapters fine-tuned for different purposes such as coding, or SQL using Hugging Face's Text Generation Inference (TGI) Deep Learning Container (DLC) in combination with a custom handler on Vertex AI.

### Fine-tuning

- [fine-tuning/vertex_ai_trl_fine_tuning_gemma.ipynb](./fine-tuning/vertex_ai_trl_fine_tuning_gemma.ipynb) - This notebooks provides steps and code to fine-tune Google Gemma with TRL via the Hugging Face PyTorch DLC for Training on Vertex AI.

### Evaluation

- [evaluation/vertex_ai_tgi_gemma_with_genai_evaluation.ipynb](./evaluation/vertex_ai_tgi_gemma_with_genai_evaluation.ipynb) - This notebooks provides steps and code to use the Vertex AI Gen AI Evaluation framework to evaluate Gemma 2 in a summarization task.
- [evaluation/vertex_ai_tgi_evaluate_llm_with_open_judge.ipynb](./evaluation/vertex_ai_tgi_evaluate_llm_with_open_judge.ipynb) - This notebooks shows how to use custom judge model to evaluate LLM-based application using the Autorater configuration in Gen AI Eval service.

### Use cases

- [use-cases/bigquery_ml_llama_inference.ipynb](./use-cases/bigquery_ml_llama_inference.ipynb) - This notebook showcases a simple end-to-end process for extracting entities and performing data analytics using BigQuery in conjunction with an open-source text-generation Large Language Model (LLM). We use Meta's Llama 3.3 70B model as an example.
- [use-cases/cloud_run_ollama_gemma2_rag_qa.ipynb](./use-cases/cloud_run_ollama_gemma2_rag_qa.ipynb) - This notebooks provides steps and code to deploy an open source RAG pipeline to Cloud Run using Ollama and the Gemma 2 model.
- [use-cases/guess_app.ipynb](./use-cases/guess_app.ipynb) - This notebook shows how to build a "Guess Who or What" app using FLUX and Gemini.
- [vertex_ai_deepseek_smolagents.ipynb](./use-cases/vertex_ai_deepseek_smolagents.ipynb) - This notebook showcases how to deploy DeepSeek R1 Distill Qwen 7B from the Hugging Face Hub on Vertex AI using Vertex AI Model Garden. It also shows how to prototype and deploy a simple agent using HuggingFace's smol-agents library on Vertex AI Reasoning Engine.


### Hyper Parameter Tuning
- [Distributed Hyperparameter Tuning](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/hyperparameter_tuning/distributed-hyperparameter-tuning.ipynb)

### ETC
- [Get started with Cloud Deploy Vertex AI Model Deployer](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_registry/get_started_with_vertex_ai_deployer.ipynb)
- [Determining the ideal machine type to use for Vertex AI endpoints](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/find_ideal_machine_type/find_ideal_machine_type.ipynb)
- [nsembling Multiple Models with NVIDIA Triton Inference Server with a custom container](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/nvidia-triton/get_started_with_triton_ensemble.ipynb)