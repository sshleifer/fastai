import os


# http://forums.fast.ai/t/send-yourself-a-text-when-training-is-complete/5256
# pip install twilio
try:
    TWILIO_TOK, MY_NUM, TWILIO_ID = os.environ['TWILIO_TOK'], os.environ['MY_NUM'], os.environ['TWILIO_ID']
except Exception:
    TWILIO_TOK, MY_NUM, TWILIO_ID = (None, None, None)

def send_sms(
        message, to=MY_NUM, tok=TWILIO_TOK, id=TWILIO_ID, twilio_num='+15085440501'):
    """Send a text message """
    from twilio.rest import Client
    if to is None: raise KeyError('need TWILIO_TOK, MY_NUM and TWILIO_ID in environment')
    client = Client(id, tok)
    client.messages.create(from_=twilio_num, to=to, body=message)
    print(f'Sent text to {MY_NUM}')

def try_send_sms(*args, **kwargs):
    try:
        send_sms(*args, **kwargs)
    except Exception:
        pass

if __name__ == '__main__':
    send_sms('testing from command line')
