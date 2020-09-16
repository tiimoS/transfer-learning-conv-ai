# Emotional Reaction Predictor (ERP) - Predicting the Emotional State of Users in Conversational Agents.


## Transformers Installation

Based on the github project by huggingface: https://github.com/huggingface/transformers

#### Setup this repo 
1. Install requirements
``
pip3 install -r requirements.txt
``

#### Setup Transformers: 
1. Clone huggingface repo
2. Install TensorFlow 2.0 and PyTorch 
3. Install Transformers using
``
pip3 install transformers
``
4. Install pyTest to verify packages: 
``
pip3 install pytest
``
5. Verify correct setup by running test-scripts: 
```
python -m pytest -sv ./transformers/tests/
python -m pytest -sv ./examples/

```

## Setup Conversational AI with Transfer Learning
1. Setup huggingface repo similar to Transformers above: https://github.com/huggingface/transfer-learning-conv-ai.git 
2. Install dependencies: 

```
cd transfer-learning-conv-ai
pip3 install -r requirements.txt
```

3.Install spacy and download english inference scripts
```
pip3 install spacy
python -m spacy download en
```

### Pretrained Model
4. Download pre-trained and fine-tuned model (optional):
[finetuned_chatbot_gpt](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz)

5. Download Script and cache model and run sample bot: 
```
python3 interact.py
```

#### Run Chatbot in telegram
1. Run the telegram_bot_integration.py file
``
python3 telegram_bot_integration.py
``

2. Open telegram and start chatting with new bot @transformer_openai_bot

## Git Triangular workflow Setup
1. Fork Repository to be included in project (remote upstream)
2. Add forked repo as submodule to existing repo: 
``
git submodule add git://github.com/tiimoS/transformers.git transformers
cd transformers
``
2. Setup forked repo upstreams: 
``
git config remote.pushdefault origin
git config push.default current
``
3. Create upstream to original project: 
``
git remote add upstream https://github.com/huggingface/transformers.git
git fetch upstream
``

