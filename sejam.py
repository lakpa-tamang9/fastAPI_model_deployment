import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from nltk import tokenize as tok

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./resources/intents_ko_yg_64sent.json', 'r') as json_data:
    intents = json.load(json_data)

# FILE = "./models/model.pth"
FILE = "./models/ko_yg_model_64sent.pth"
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

# print(all_words)
# sentence = "do you use credit cards?"
name = input("Enter your name: ")
bday = int(input("Please enter your age !: "))
gender = input("Please identify your gender!: ")
place = input("Where are you from ?: ")

personal_infos = (" ").join([name, str(bday), gender, place])

personal_info = [name, str(bday), gender, place]
print(personal_info)

sentence = tokenize(personal_infos)

print(sentence)

X = bag_of_words(sentence, all_words)
# return bag of words array: 1 for each known word that exists in the sentence, \ 
# 0 otherwise example: sentence = ["hello", "how", "are", "you"] words = ["hi", "hello", "I", "you", "bye", "thank", "cool"] bog = [ 0 , 1 , 0 , 1 , 0 , 0 , 0]
X = X.reshape(1, X.shape[0])
X = torch.from_numpy(X).to(device)

output = model(X)
_, predicted = torch.max(output, dim=1)

tag = tags[predicted.item()]

probs = torch.softmax(output, dim=1)
prob = probs[0][predicted.item()]
# print(prob.item())
# if prob.item() > 0.75:
for intent in intents['intents']:
    if tag == intent["tag"]:
        print(f"Keywords: {random.choice(intent['keywords'])}")
        keywords = random.choice(intent['keywords'])

#Algorithm for selecting the sentences.
# Load the sentences from fortune lake json

with open("./resources/fortune_ko_yg_64sent.txt", "r") as f:
    fortune_sentences = f.read()

fortune = tok.sent_tokenize(fortune_sentences)
final_fortune = []
for fort in fortune:
    for keyword in keywords:
        if keyword in fort:
            final_fortune.append(fort)

final_list = (" ").join(sentence for sentence in set(final_fortune))
print(final_list)
    