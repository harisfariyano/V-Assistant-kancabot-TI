import random # library digunakan untuk menghasilkan angka acak di Python.
import json #library untuk menghasilkan file json
import torch
# import app
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
# from flask_mysqldb import MySQL
# import speech_v1 as speech


# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'flask'
# mysql = MySQL(app)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#membuka file json
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
#membuat fungsi response
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    #membuat jawaban yang tidak diketahui
    return 'Maaf saya tidak mengerti yang anda maksud'

# def speech_to_text(config, audio):
#     client = speech.SpeechClient()
#     response = client.recognize(config=config, audio=audio)
#     print_sentences(response)

# def get_response(msg):
#     import speech_recognition as sr
#     ear = sr.Recognizer()
#     with sr.Microphone() as sourse:
#         print("listening...")
#         audio = ear.listen(sourse)
#         try:
#             text = ear.recognize_google(audio)
#             print(text)
#         except:
#             print("i didn't get that...")
#             resp= get_response(sentence)
#             print(resp)


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
        # cursor = mysql.connection.cursor()
        # cursor.execute(''' INSERT INTO test VALUES(NULL,%s,%s,NULL)''',(sentence,resp))
        # mysql.connection.commit()
        # cursor.close()
