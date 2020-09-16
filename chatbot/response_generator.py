# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, cached_path

import random
import warnings
import logging
import os
import tarfile
import torch
import torch.nn.functional as F

import empathy_enhancer

work_dir = os.getcwd()
FINE_TUNED_MODEL = work_dir + '/../models-fine-tuned/language_model_empathic.tar.gz'

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ('<speaker1>', '<speaker2>')}

logger = logging.getLogger(__file__)

tokenizer = None
model = None
history = []
args = None


def init():
    global args, tokenizer, model, logger
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=2, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.9, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info(pformat(args))

    args.model_checkpoint = download_pretrained_model()
    checkpoint = args.model_checkpoint + 'output/'

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info('Get pre-trained model and tokenizer...')
    tokenizer_class, model_class = OpenAIGPTTokenizer, OpenAIGPTLMHeadModel

    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    add_special_tokens_()

    logger.info('Ready to talk!')


def download_pretrained_model():
    """
    Downloads the pre-trained language model and loads it as the model.
    :return: path to the language model
    """
    logger.info('Downloading pretrained model...')
    resolved_archive_file = cached_path(FINE_TUNED_MODEL)
    path = os.getcwd() + '/code/models-fine-tuned/'
    logger.info('Extracting archive file {} to temp dir {}'.format(resolved_archive_file, path))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(path)
    return path


def add_special_tokens_():
    """
    Add special tokens to the tokenizer and the model if they have not already been added.
    """
    global model, tokenizer
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def generate_response(num_samples=5, repetition_penalty=1.2, tokenized_message=None):
    """
    Generates multiple (num_samples) possible responses to given input. It does so by generating the words in the
    responses token (word) by token, for all possible responses in parallel. Then it applies a repetition penalty and
    softmax to get the prediction probabilities and picks the best token to attach to the sequence of already generated
    tokens of each reply. This process is continued until every possible response ends with an end of sequence token
    <eos>.
    :param num_samples: number of response candidates to generate
    :param repetition_penalty: penalty to apply for repeating tokens in sequence generation, best value is 1.2
    :return: list of num_samples generated replies
    """
    global args, model, tokenizer, history

    # Prepare tmp array to hold replies during response generation
    replies = []
    for i in range(num_samples):
        replies.append([])

    tmp_history = history.copy()
    if tokenized_message:
        tmp_history.append(tokenized_message)

    not_finished = True  # is false if all sentences being generated end with the end of sequence token <eos>

    with torch.no_grad():
        for idx in range(args.max_length):
            if not_finished:
                # Build the sequence of input tokens and token type ids to be used as model inputs
                input_tokens, token_type_ids = build_model_input(replies, tmp_history=tmp_history)

                # Calculate raw response token predictions by feeding the multi-dimensional input to the model.
                logits = model(input_tokens, token_type_ids=token_type_ids)[0]

                # Only use the predictions for the next token and use temperature to control the randomness of the pred.
                next_token_logits = logits[:, -1, :] / (args.temperature if args.temperature > 0 else 1.)

                # Apply repetition penalty and next token generation for each response prediction sample
                for i, sample in enumerate(input_tokens):
                    # Only generate new token if the existing reply is not already finished generating
                    if len(replies[i]) == 0 or replies[i][-1] != tokenizer.encode('<eos>')[0]:
                        # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858), best at 1.2
                        for _ in set(sample.view(-1).tolist()):
                            next_token_logits[i][_] /= repetition_penalty

                        next_token = calculate_next_token(i, idx, next_token_logits)
                        replies[i].append(next_token.item())
                    else:
                        # Add the end-of-sequence token to replies that already finished to ensure that all possible
                        # replies still have the same length for the model to process them
                        replies[i].append(tokenizer.encode('<eos>')[0])

                not_finished = are_replies_finished(replies)
            else:
                # stop response generation if all replies end with end-of-seq token.
                break
        return replies


def build_model_input(replies, tmp_history):
    """
    Builds an input sequence of tokens for the model based on the bots personality, conversation history, and
    reply tokens generated so far. Also generates a sequence of token types that assign each input token to a speaker,
    so its a positional embedding defining which speaker said which word in the input sequence. Both sequences need to
    have the same length. Generates such sequences for all possible reply candidates.
    :param replies: list of possible replies currently being generated with each reply being a sequence of tokens
    :param tmp_history: conversation history as list of tokenized messages
    :return: multi-dimensional tensor with each row representing input tokens for single response candidate.
    In addition, returns the longest sequence of token type ids i.e. speaker-input token assignments to be used as
    model input too.
    """
    token_types = []
    inputs = []
    for i in range(len(replies)):
        token_types.append([])
        inputs.append([])

    # Build the model input as a sequence of tokens containing the bots personality,
    # conversation history with emotion labels and generated reply so far.
    for k, reply in enumerate(replies):
        instance = build_input_from_segments(reply, tmp_history)

        # sequence of input tokens for the model as described above
        input_ids = instance["input_ids"]

        # position/speaker embedding, i.e. assigns token at index in input_ids either to speaker1 or
        # speaker2, depending on who said it.
        speaker_embedding = instance["token_type_ids"]

        inputs[k] = input_ids
        token_types[k] = speaker_embedding

    # Generate a multi-dimensional tensor with each row representing a possible input for the model
    generated = torch.tensor(inputs, dtype=torch.long, device=args.device)
    token_type_ids = find_max_token_type_ids(token_types)
    return generated, token_type_ids


def build_input_from_segments(reply, tmp_history=None):
    """
    Builds the input for the language model from the personality segment, the conversation history and the
    generated reply (so far). All words in the sequence are tokenized to ids according to the vocabulary.
    :param reply: the so far generated sequence of tokens for the reply
    :param tmp_history: history with a potential response attached, used for predicting the user reaction to the
    response. It is undefined for normal response generation.
    :return: The input sequence of tokens containing personality, history and reply. In addition returns token type ids
    that assign each input token to a speaker.
    """
    global tokenizer

    # Build a sequence of input from 3 segments: persona, history and last reply.
    personality = []
    # personality = [tokenizer.encode("I am a chatbot"), tokenizer.encode("My name is Joy")]
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*personality))] + tmp_history + [reply + []]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1]
                                + s for i, s in enumerate(sequence[1:])]
    instance = dict()
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]

    return instance


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def are_replies_finished(replies):
    """
    Checks if all generated replies end with an end of sequence token. If so, the generation is being completed
    :param replies: list of reply candidates currently being generated with each reply being a sequence of tokens
    :return: boolean - false if there is still a reply that is under generation, true if all replies are completed
    """
    # Check if all reply candidates have finished generating i.e. end with the end of sequence token
    not_finished = False
    for reply in replies:
        if reply[-1] != tokenizer.encode('<eos>')[0]:
            not_finished = True
    return not_finished


def calculate_next_token(i, idx, next_token_logits):
    """
    Calculate the next token id from the raw predictions (logits) by first filtering the predictions using nucleus
    or threshold filtering, then calculating the softmax of the predictions to get the probabilities of each possible
    token and last getting the max prob. token of this list.
    If the next token is a special token or
    :param i: running index for response candidate for which we are currently generating tokens, in range 0-num. samples
    :param idx: running index for token in output sequence. in range 0 to max-length of response (=80)
    :param next_token_logits: tensor for raw predictions of next token. contains predictions for all response candidates
    :param special_tokens_ids: token ids of special tokens such as end of sequence <eos> or beginning of sequece <bos>
    :return: next word (as token id) to be attached to the reply of response candidate i
    """
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    # Filter raw predictions using nucleus or threshold filtering
    filtered_logits = top_filtering(next_token_logits[i], top_k=args.top_k, top_p=args.top_p)
    if args.temperature == 0:  # greedy sampling:
        next_token = torch.argmax(filtered_logits)
    else:
        # Generate probabilities and tokenized word for next token
        next_token_probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1)

        # continue generating new tokens as long as the model produces only special tokens as outputs and min. length
        # of output sequence is not reached.
        if idx < args.min_length and next_token.item() in special_tokens_ids:
            while next_token.item() in special_tokens_ids:
                if next_token_probs.max().item() == 1:
                    warnings.warn("Warning: model generating special tokens with probability 1.")
                    break
                next_token = torch.multinomial(next_token_probs, num_samples=1)
    return next_token


def find_max_token_type_ids(token_types):
    """
    Finds the longest sequence of token type ids aligning with the longest generated response so far.
    We use the token_type_ids of the longest generated sequence, since the input length has to be the same
    as the token type ids length in order to assign each input word to a speaker.
    :param token_types: token_type_ids for each response under generation with the same length as response
    :return: longest token_type_id sequence, to be used as input for the model.
    """
    max_len = 0
    for token_type in token_types:
        if len(token_type) > max_len:
            token_type_ids = token_type
            max_len = len(token_type)
    token_type_ids = torch.tensor(token_type_ids, device=args.device).unsqueeze(0)
    return token_type_ids


def decode_texts(replies):
    """
    Decodes the tokenized replies back to words and removes the end of sequence tokens from the reply.
    :param replies: list of tokenized replies
    :return: list of replies in text format and list of tokenized replies without <eos> tokens
    """
    eos_token = tokenizer.encode('<eos>')[0]
    texts = []
    output_ids = []
    for out in replies:
        cleaned_out = []
        for token in out:
            if token != eos_token:
                cleaned_out.append(token)
        output_ids.append(cleaned_out)
        text = tokenizer.decode(cleaned_out, clean_up_tokenization_spaces=False, skip_special_tokens=True)
        texts.append(text)
    return texts, output_ids


def add_to_history(message):
    global history, tokenizer
    message_ids = tokenizer.encode(message)
    history.append(message_ids)


def tokenise_msg(labelled_msg, num_samples=5):
    """
    Tokenize message according to vocabulary to a list of ids / tokens
    :param labelled_msg: user message with emotion label(s) at the end.
    :param num_samples: number of response candidates to generate
    :return: tokenised text and ids.
    """
    message = tokenizer.encode(labelled_msg)
    history.append(message)
    outputs = generate_response(num_samples=num_samples, repetition_penalty=1.2)
    return decode_texts(outputs)


def get_response(user_message, num_samples=5, reset=False):
    """
    Generates a response to the user message. First, detects the emotion of the message and attaches it to the text.
    Then it generates multiple response candidates. For each candidate it predicts the users emotional reaction to it,
    by classifying the emotion of the predicted user response to the candidate response. The response with the highest
    prediction probability for the joyful emotion is being selected and returned as the response to the user.
    :param user_message: text message from telegram api
    :param reset: if the conversation history should start from scratch e.g. after re-starting the conversation
    :param num_samples: number of response candidates to generate
    :return: response to user_message as text
    """
    global history, tokenizer
    output = '****************** INTERMEDIATE RESULTS ***********************\n'
    output += 'Input message: ' + user_message + '\n'

    # Empties the conversation history when the user types /start in the telegram bot
    if reset:
        history = []

    # Predict emotion label for user message and append to the end in reversed order
    labelled_msg, labels = empathy_enhancer.inject_emotions(user_message)
    output += 'Detected User Emotion:' + ' '.join(labels) + '\n'

    # Tokenize message according to vocabulary to a list of ids / tokens
    texts, out_ids = tokenise_msg(labelled_msg, num_samples)

    # Generate response candidates and predict user reactions
    candidate_response = []
    predicted_user_responses = []
    for ids in out_ids:
        predicted_user_reaction = generate_response(num_samples=1, repetition_penalty=1.2, tokenized_message=ids)[0]
        predicted_user_response = tokenizer.decode(predicted_user_reaction[:-1],
                                                   clean_up_tokenization_spaces=False,
                                                   skip_special_tokens=True)
        predicted_user_responses.append(predicted_user_response)
    batches = empathy_enhancer.detect_emotion(predicted_user_responses)

    if len(batches) == 0:
        print('Batches are empty')
        candidate_response.append((predicted_user_responses[0], 1))

    # Select response with highest joyful probability
    candidate_response, output = empathy_enhancer.select_best_response_candidate(batches, candidate_response, output,
                                                                                 predicted_user_responses, texts)

    text, prob = max(candidate_response, key=lambda item: item[1])

    # Append response to history
    response_ids = out_ids[texts.index(text)]
    history.append(response_ids)
    history = history[-(2 * args.max_history + 1):]
    output += '***************************************************************\n'
    return text, output


def main():
    init()
    while True:
        raw_text = input(">>> ")
        response, process_output = get_response(raw_text)
        print(process_output)
        print(response)


if __name__ == "__main__":
    main()
