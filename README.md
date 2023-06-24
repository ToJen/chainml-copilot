# chainml-copilot
Our ETHWaterloo 2023 hackathon project is an AI agent that simplifies user onboarding and participation in web3 communities of their choice.

> NOTE: This project makes use of a publicly available [ChainML model](https://github.com/chain-ml/tmls-2023-material).

## Requirements

- python3
- pip3

## Setup

### Model Server

- `$ pip3 install -r requirements.txt`
- Create a new file called *config.env*
    - Copy the contents of *config.env.example* into your new *config.env* file
    - API keys for third party tools are not provided.
        - `OPENAI_API_KEY` from [OpenAI](https://platform.openai.com/account/api-keys)
        - `GOOGLE_API_KEY` from [Google News Search](https://programmablesearchengine.google.com/controlpanel/all)
        - `CX` from [Google Search](https://console.developers.google.com/)
            - _use the search id_

## Usage

- `$ python model_server.py`

