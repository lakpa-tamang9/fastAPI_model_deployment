
import random
import json

import torch
from pydantic import BaseModel
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from nltk import tokenize as tok

from fastapi import FastAPI

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Score(BaseModel):
    name: str
    age: int
    place: str
    gender: str

def load_intents():
    with open('./resources/intents.json', 'r') as json_data:
        return json.load(json_data)

def load_txt():
    with open("./resources/fortune_lake.txt", "r") as f:
        return f.read()

def load_model(file):
    data = torch.load(file)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, all_words, tags

def input_processing(personal_info, all_words):
    sentence = tokenize(personal_info)
    X = bag_of_words(sentence, all_words)
    # return bag of words array: 1 for each known word that exists in the sentence, \ 
    # 0 otherwise example: sentence = ["hello", "how", "are", "you"] words = ["hi", "hello", "I", "you", "bye", "thank", "cool"] bog = [ 0 , 1 , 0 , 1 , 0 , 0 , 0]
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    return X

def search_sentence(fortune_sentences, keywords):
    final_fortune = []
    fortune = tok.sent_tokenize(fortune_sentences)
    for fort in fortune:
        for keyword in keywords:
            if keyword in fort:
                final_fortune.append(fort)
    return set(final_fortune)

@app.get("/")
async def predict(item: Score):
    model, all_words, tags = load_model("./models/model.pth")
    intents = load_intents()
    fortune_sentences = load_txt()
    request_json = item.dict()
    data_list = list(request_json.values())
    concat_input = (" ").join([str(val) for val in data_list])
    input_data = input_processing(concat_input, all_words)
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]
    # print(prob.item())
        # if prob.item() > 0.75:

    for intent in intents['intents']:
        if tag == intent["tag"]:
            print(f"Keywords: {random.choice(intent['responses'])}")
            keywords = random.choice(intent['responses'])
    final_fortune = search_sentence(fortune_sentences=fortune_sentences, keywords=keywords)
    return {"keywords": keywords, "fortune": final_fortune}

