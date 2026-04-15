# wxai-deployable-ai-services-and-functions

A toolkit for developing, uploading, and deploying [watsonx.ai Deployable Python Functions](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-deploy-py-function-write.html) and [Deployable AI Services](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ai-services-manual-coding.html) on IBM watsonx.ai.

---

## Repository Structure

```text
.
├── src/                        # Upload and deployment utilities
│   ├── utils/                  # Core workflow modules
│   └── helpers/                # Authentication helpers
├── deployable_functions/       # Deployable Python Function templates and examples
├── deployable_ai_services/     # Deployable AI Service templates and examples
├── documentation/              # Reference docs from watsonx.ai and the ibm-watsonx-ai Python SDK
└── config/                     # Environment variable templates
```

---

## src/

The `src/` directory contains Python modules that handle the full upload-and-deploy lifecycle. Functions and AI services are uploaded as assets first, then deployed as online or batch endpoints.

### src/utils/

| File                        | Purpose                                                                                                                                                                                                         |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wxai_functions_upload.py`  | Uploads a deployable Python function to watsonx.ai. Auto-discovers the `score` function, input/output JSON schemas, and `requirements.txt`. Creates a custom software spec with package extensions when needed. |
| `wxai_functions_deploy.py`  | Deploys an uploaded function asset as an online or batch deployment. Auto-derives the deployment name from the asset if not provided.                                                                           |
| `wxai_ai_service_upload.py` | Uploads a deployable AI service. Auto-discovers which of `generate`, `generate_stream`, and `generate_batch` are implemented. Parses the AST to verify functions are non-trivial.                               |
| `wxai_ai_service_deploy.py` | Deploys an uploaded AI service asset. Online deployments expose `generate` and `generate_stream`; batch deployments expose `generate_batch`.                                                                    |
| `load_all_dotenv.py`        | Loads `.env` files from a directory or a specific file. Useful for managing credentials across multiple environments.                                                                                           |

### src/helpers/

| File                       | Purpose                                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auth_helper_functions.py` | IBM Cloud authentication utilities - IAM token retrieval (`get_iam_token`, `auth_iam_token`) and Zen API header generation (`generate_zen_auth_header`). |

---

## Deployable Python Functions

Located in `deployable_functions/`. Each subdirectory contains a self-contained function with its Python file, input/output JSON schemas, and a `requirements.txt`.

**Folders with `template` in the name are starter templates** - copy and customise them for your own use case.  
**All other folders are example implementations** you can reference or deploy directly.

| Folder                          | Type     | Description                                                                                                                                                                                     |
| ------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deployable_function_template/` | Template | Generic starter template. Includes a `FunctionState` class for environment variable management, runtime override support, and helper functions for parsing payloads and building responses.     |
| `migrate_cos_buckets/`          | Example  | Example function for transferring files between two IBM Cloud Object Storage (COS) instances. Accepts source/target COS credentials and optional object prefix filters via the request payload. |
| `function_runtime_debugger/`    | Example  | Inspects the deployed runtime - lists installed packages and versions, and can pip-install or upgrade packages on demand. Useful for troubleshooting custom software specs.                     |

---

## Deployable AI Services

Located in `deployable_ai_services/`. Each subdirectory contains a self-contained AI service with its Python file, request/response JSON schemas, and a `requirements.txt`.

**Folders with `template` in the name are starter templates** - copy and customise them for your own use case.  
**All other folders are example implementations** you can reference or deploy directly.

| Folder                            | Type     | Description                                                                                                                                                                                                                                                      |
| --------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deployable_ai_service_template/` | Template | Generic starter template. Implements `generate`, `generate_stream`, and `generate_batch`. Includes a `ServiceState` class for environment variable management, context-based token generation, and helper functions for parsing requests and building responses. |
| `ai_service_runtime_debugger/`    | Example  | Inspects the deployed runtime - lists installed packages and versions, and can pip-install or upgrade packages on demand. Supports streaming via `generate_stream`.                                                                                              |

---

## Documentation

Located in `documentation/`. These are reference documents pulled from the official watsonx.ai documentation pages and the `ibm-watsonx-ai` Python SDK docs.

| Folder                              | Source                                                                                                               | Contents                                                                                                                                                              |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wxai_deployable_python_functions/` | [watsonx.ai docs](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-deploy-py-function-write.html) | Writing deployable Python functions - `score` function requirements, input/output format, testing and deployment.                                                     |
| `wxai_deployable_ai_services/`      | [watsonx.ai docs](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ai-services-manual-coding.html)   | Writing deployable AI services - `generate`, `generate_stream`, and `generate_batch` definitions, context management, REST API endpoints, online vs. batch behaviour. |
| `wxai_deployment_docs/`             | ibm-watsonx-ai Python SDK docs                                                                                       | SDK reference for deployments and parameter sets, plus online and batch deployment documentation.                                                                     |

---

## Configuration

Copy the templates in `config/TEMPLATES/` and fill in your credentials:

| Template                 | Purpose                                                             |
| ------------------------ | ------------------------------------------------------------------- |
| `_ibmcloud.env.TEMPLATE` | IBM Cloud API key and Secrets Manager configuration                 |
| `_wxai.env.TEMPLATE`     | watsonx.ai API key, Space ID, deployment URL, and default model IDs |

---

## Requirements

- Python 3.12+
- Key dependencies: `ibm-watsonx-ai`, `python-dotenv`, `requests`

Install all dependencies with:

```bash
uv sync
# or
uv add install -r requirements.txt
```
