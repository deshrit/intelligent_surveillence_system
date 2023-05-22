# intelligent_surveillence_system
A terible multi-threaded pyqt5 application that uses custom trained yolov4-tiny model to detect 4 different classes of weapons and sends email and sms about first detection, email includes images.

## UI
![test](https://github.com/deshrit/intelligent_surveillence_system/assets/59757711/3747b851-8e5a-4986-8570-d65edf6ba1ba)


*`Restart` button hasnot been programmed*

## To run locally
1. Create a virtual enviromnet and run `pip install -r requirements.txt`
2. Add `.env` file on root of project and add following on files
```
# email
EMAIL_AUTH_TOKEN="your_token"
# SMS
SMS_ACCOUNT_SID="your_twilio_sid"
SMS_AUTH_TOKEN="your_twilio_auth_token"
```
10. Update `credentials.json` to add following
```
"EMAIL_SENDER": "your_email_sender_account",
"EMAIL_RECEIVERS": [
  "email_1",
  "email_2",
]

"SMS_SENDER":  "your_sender_twilio_number",
"SMS_TARGETS": ["receiver_1",]
```
3. Make directory to stored detected images `mkdir images`
4. Run `python app.py`
