import subprocess

PYTHON_PATH = '/Users/timospring/.pyenv/versions/2.7.17/envs/deepmoji/bin/python'
CLASSIFIER_PATH = '/Users/timospring/Desktop/DeepMoji/emotion_classifier.py'


def inject_emotions(user_message):
    """
    Runs the emotion detection on the user message and appends it as emotion labels to the msg
    :param user_message: text message from user
    :return: labelled user message and emotion labels used
    """
    emotion_predictions = detect_emotion([user_message])
    labels = []
    for label_batch in emotion_predictions:
        label = label_batch[0][0]
        emotion_label = '#' + label
        labels.append(emotion_label)
    labels.reverse()
    labelled_msg = user_message + ' ' + ' '.join(labels)

    return labelled_msg, labels


def detect_emotion(messages_pred):
    """
    Predicts the emotion label for a given list of messages using the modified DeepMoji emotion classifier.
    Since the classifier runs in a python 2.7 environment, it has to be called using sub-processes.
    :param messages_pred: list of messages to predict emotions for
    :return: list of emotion labels for each input message
    """
    # Make classifier run with different python environment i.e. python 2.7
    emotion_pred_process = subprocess.Popen([PYTHON_PATH, CLASSIFIER_PATH],
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE)

    # Build single input line from all messages and delimit with *
    messages = ''
    for message in messages_pred:
        messages += message + '*'

    # Predict emotion labels for all input sentences
    out, err = emotion_pred_process.communicate(str.encode(messages))
    out_str = out.decode('utf-8')
    emotion_pred_process.terminate()

    # Splits the output back to individual predictions for each message
    prediction_batches = out_str.splitlines()[-1].split('|')[:-1]
    batches = []
    for batch in prediction_batches:
        emotion_labels = batch.split(';')[:-1]
        emotion_labels.reverse()
        emotions = []
        probs = []
        for label in emotion_labels:
            emotion, prob = label.split('-')
            emotions.append(emotion)
            probs.append(float(prob))
        batches.append((emotions, probs))
    return batches



def select_best_response_candidate(batches, candidate_response, output, predicted_user_responses, texts):
    """
    Selects the response candidate that causes a joyful reaction with the highest prediction probability.
    :param batches: batches of emotion labels
    :param candidate_response: response candidates to be considered
    :param output: output of the selection to be printed in the console.
    :param predicted_user_responses: predicted user responses
    :param texts: candidate response in text format
    :return: the output to be printed in the console and the candidates response to be returned
    """
    output += 'Possible Responses and user reactions to them:\n'
    for i, batch in enumerate(batches):
        output += '\t' + str(i) + '. ' + texts[i] + '\n'
        labels = batch[0]
        probs = batch[1]
        label_string = ''
        for idx, label in enumerate(labels):
            label_string += label + ' ' + str(probs[idx]) + '; '
        output += '\t\t User reaction: ' + predicted_user_responses[i] \
                  + '\t\t Emotion Labels: ' + label_string + '\n'

        if 'joyful' in batch[0]:
            index = batch[0].index('joyful')
            candidate_response.append((texts[i], batch[1][index]))
    if len(candidate_response) == 0:
        candidate_response.append(texts[0])

    return candidate_response, output

