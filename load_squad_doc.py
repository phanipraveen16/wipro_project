import os
import json
import requests
from load_from_wiki import load_data

def get_documents(dataset):
    raw_data = []
    documents = []
    for data in dataset['data']:
        document = {}
        document["title"] = data["title"]
        document["passages"] = []
        paragraphs = data['paragraphs']
        for paragraph in paragraphs:
            para_ques_dict = {}
            para_ques_dict['context'] = paragraph['context']
            ques_list = []
            for questions in paragraph['qas']:
                ques_list.append(questions['question'])
            para_ques_dict['questions'] = list(set(ques_list)) 
            document["passages"].append(para_ques_dict)
        documents.append(document)
    return documents


def process_squad_data(in_file_path, out_file_path):
    with open(in_file_path) as data_file:
        dataset = json.load(data_file)
        documents = get_documents(dataset)

    with open(out_file_path, 'w') as outfile:
        json.dump(documents , outfile)
      
if not os.path.isfile('../data/squad_train_data_doc.json'):
    if not os.path.isfile("../data/train-v1.1.json"):
        print("Loading Squad Training Data")
        response = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
        with open("../data/train-v1.1.json", "wb") as outfile:
            for data in response.iter_content():
                outfile.write(data)
        process_squad_data("../data/train-v1.1.json", "../data/squad_train_doc.json")

# Check if the dev-v1.1.json exists
if not os.path.isfile('../data/squad_dev_data_doc.json'):
    if not os.path.isfile("../data/dev-v1.1.json"):
        print("Loading Squad Dev Data")
        response = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
        with open("../data/dev-v1.1.json", "wb") as outfile:
            for data in response.iter_content():
                outfile.write(data)
        process_squad_data("../data/dev-v1.1.json", "../data/squad_dev_doc.json")