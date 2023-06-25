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

### XMTP Server

- Create a new file called *.env*
    - Copy the contents of *.env.example* into your new *.env* file
        - `KEY` refers to your Ethereum private key


## Usage

### Model Server

- `$ python model_server.py`
  - This runs on **http://localhost:5555** and the chatbot endpoint is **http://localhost:5555/chat**

#### Chat API Schema

##### Sample Request
```json
{
    "prompt": "What is the difference between uniswap v2 and uniswap v3?"
}
```
##### Sample Response
```json
{
    "controller_response": "1;uni_investor_material;Uniswap v2 vs v3 differences\n2;google_news_search;Uniswap v2 and v3 comparison",
    "fact_check_passed": true,
    "fact_checking_response": "Step 1: Identify the differences mentioned in the AI-Generated Response.\n- ... Handling Non-Standard Tokens\n- Protocol Fees\n- Language and Architectu...the AI-Generated Response.\n- All the differences mentioned in the AI-Generated Response are supported by the Retrieved Context.\n\nTrue",
    "generated_query": "Uniswap v2 vs v3 differences",
    "retrieved_context": "As mentioned above, Uniswap v2 stores the last recorded balance of each asset (in order to\nprevent a particular manipulative exploit of the oracle mechanism). The new architecture...Uniswap v2 is to minimize the surface area and complexity of\n\n",
    "selected_skill": "uni_investor_material",
    "user_message": "What is the difference between uniswap v2 and uniswap v3?",
    "web3_devrel_response": "The main differences between Uniswap v2 and Uniswap v3 are as follows:\n\n1., ... more advanced and flexible version of the protocol, offering improved capital efficiency, better fee structures, and enhanced price oracle functionality."
}
```

### XMTP Server

- `$ npm i`
- `$ npm start`

# Adding more docs from web3 projects

- Scrape and download the project's entire documentation site: `$ wget --limit-rate=200k --no-clobber --convert-links --random-wait -r -p -E -e robots=off -U mozilla http://www.SOME_WEBSITE.com`
- Make a copy of the newly created *www.SOME_WEBSITE.com/* directory (likely deeply nested): `$ cp -R www.SOME_WEBSITE.com temp_copy_SOME_WEBSITE`
- Flatten the cloned directory to one level: `$ find . -mindepth 2 -type f -exec mv -if '{}' . ';'`
- Convert all to a single PDF
- Place the final PDF output in the `data` directory in the `model_server`
- Update `model_server.py` accordingly (in the future this will be automated but is currently manual).

# Future Work

- Provide wallet data in requests
- Parse any "legal documents" or policies that individual DAOs may have and allow users to ask questions about the Terms & Conditions as opposed to the default DAO documentation.


# Screenshots
See [./screenshots](./screenshots).
