import pyrebase

config = {
    'apiKey': "AIzaSyBixtA4v5mvxKvaTU61iq9Fr2Ln2OWlf3o",
    'authDomain': "tomatocare-78e23.firebaseapp.com",
    'projectId': "tomatocare-78e23",
    'storageBucket': "tomatocare-78e23.appspot.com",
    'messagingSenderId': "437959910172",
    'appId': "1:437959910172:web:dab8c80225929289dd90d9",
    'databaseURL' : "",
  }

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

email = 'test@gmail.com'
password = '123456'

user = auth.sign_in_with_email_and_password(email, password)
#info = auth.get_account_info(user['idToken'])
#print(info)

auth.send_password_reset_email(email)
#updated