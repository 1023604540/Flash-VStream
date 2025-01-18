import os
import json
import ast
from collections import defaultdict
from openai import OpenAI

class ScoreMeter:
    def __init__(self):
        self.score_sum = 0
        self.count = 0
        self.yes_count = 0
        self.no_count = 0
        self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

    def add_score(self, score, pred):
        self.score_sum += score
        self.count += 1
        pred_lower = pred.lower()
        if 'yes' in pred_lower:
            self.yes_count += 1
            self.score_dict['yes'][score] += 1
        elif 'no' in pred_lower:
            self.no_count += 1
            self.score_dict['no'][score] += 1

    def get_average_score(self):
        res = (self.score_sum / self.count) if self.count else 0
        return f"{res:.6f}"

    def get_accuracy(self, response_type):
        if response_type == 'yes':
            res = (self.yes_count / self.count) if self.count else 0
        elif response_type == 'no':
            res = (self.no_count / self.count) if self.count else 0
        else:
            res = 0
        return f"{res:.6f}"

def retrieve_results(output_dir, api_key):
    client = OpenAI(api_key=api_key)
    batch_ids_file = os.path.join(output_dir, "batch_ids.txt")

    combined_contents = {}
    with open(batch_ids_file, "r") as f:
        batch_ids = f.read().splitlines()

    for batch_id in batch_ids:
        batch = client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        if output_file_id:
            file_response = client.files.content(output_file_id)
            output_file_path = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
            with open(output_file_path, "w") as output_file:
                output_file.write(file_response.text)
            with open(output_file_path, "r") as json_file:
                for line in json_file:
                    content = json.loads(line)
                    response_body = content['response']['body']
                    message_content = response_body['choices'][0]['message']['content']
                    result = ast.literal_eval(message_content)
                    assert 'pred' in result, f"Error: {output_file_path} doesn't have key=pred"
                    assert 'score' in result, f"Error: {output_file_path} doesn't have key=score"
                    combined_contents[content['custom_id']] = result

    # Combine all the processed files into one
    json_path = os.path.join(output_dir, "combined_results.json")
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Score calculation
    meter_dic = {'total': ScoreMeter()}
    for key, result in combined_contents.items():
        score = int(result['score'])
        pred = result['pred']

        meter_dic["total"].add_score(score, pred)
        if 'a_type' in result and result['a_type'] is not None:
            typ = str(result['a_type'])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, pred)

    csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}

    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    key_list = sorted([k for k in meter_dic.keys()])
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    print(output)
    output_csv = json_path.replace(".json", ".csv")
    with open(output_csv, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    output_dir = "/home/vault/b232dd/b232dd16/Flash-VStream/checkpoints-finetune/vstream-7b-finetune-weighted_kmeans1*8-25*4-25*1-STAR/checkpoint-5900/evaluation/msrvtt/test"
    api_key = "your_api_key"
    retrieve_results(output_dir, api_key)