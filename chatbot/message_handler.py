import logging
import response_generator

from telegram import ChatAction
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters)
from functools import wraps

version = '1.0'
TOKEN = 'YOUR_TOKEN'

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

num_samples = 5
show_intermediate_results = True


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return func(update, context,  *args, **kwargs)

    return command_func


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
@send_typing_action
def start(update, context):
    global num_samples, show_intermediate_results
    """Send a message when the command /start is issued."""
    update.message.reply_text('Empathic Chatbot ' + version + ' :)')
    update.message.reply_text('Chat configuration: \nShow intermediate results: ' + str(show_intermediate_results)
                              + ' (change using /show or /hide) and \nnumber of generated responses: ' + str(num_samples)
                              + ' (change using /samples {number}')
    name = update.effective_user['first_name']
    welcome_message = 'Hi ' + name + ', how are you doing?'
    response_generator.add_to_history(welcome_message)
    context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)


def help(update, context):
    """Send a message when the command /help is issued."""
    help_text = 'The following commands can be used: ' \
           '\n /start - to start a new conversation with the bot, also resetting the conversation history ' \
           '\n /samples {number} - use samples to change the number of response candidates that should be generated followed by the number to change it to' \
           '\n /show - shows the intermediate results of the response generation, ie. all response candidates, emotion detection results and user response predictions' \
           '\n /hide - hides intermediate results and only shows the generated response\n\n'
    update.message.reply_text(help_text)


@send_typing_action
def echo(update, context):
    """Calls language generation model to generate response"""
    print(context)
    update.message.reply_text(update.message.text)


@send_typing_action
def respond(update, context):
    global num_samples, show_intermediate_results
    response, process_output = response_generator.get_response(update.message.text, num_samples=num_samples)

    if show_intermediate_results:
        context.bot.send_message(chat_id=update.effective_chat.id, text=process_output)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def change_number_samples(update, context):
    global num_samples
    old_samples = num_samples
    num_samples = int(context.args[0])
    update.message.reply_text('okay, I changed the number of samples to generate from ' + str(old_samples) + ' to ' + str(num_samples))


def show_process(update, context):
    global show_intermediate_results
    show_intermediate_results = True
    update.message.reply_text('okay, I will show you the intermediate results. ')


def hide_process(update, context):
    global show_intermediate_results
    show_intermediate_results = False
    update.message.reply_text('okay, I will hide intermediate results.')


def run_bot():
    response_generator.init()
    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("samples", change_number_samples, pass_args=True))
    dp.add_handler(CommandHandler("show", show_process))
    dp.add_handler(CommandHandler("hide", hide_process))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.text, respond))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    updater.idle()


def main():
    global updater, dp
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    run_bot()


if __name__ == '__main__':
    main()
