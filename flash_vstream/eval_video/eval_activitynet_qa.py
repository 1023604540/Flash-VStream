# Based on https://github.com/haotian-liu/LLaVA.
import logging
import os
import ast
import json
from openai import OpenAI
import argparse
from tqdm import tqdm
from time import sleep
from collections import defaultdict
from multiprocessing.pool import Pool

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    parser.add_argument("--api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument("--api_type", default=None, type=str, help="OpenAI API type")
    parser.add_argument("--api_version", default=None, type=str, help="OpenAI API version")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    parser.add_argument("--batch_size", default=1000, type=int, help="Number of requests per batch file")
    args = parser.parse_args()
    return args


def prepare_batch_file(prediction_set, caption_files, output_dir, batch_size):
    batch_requests = []
    for idx, file in enumerate(tqdm(caption_files)):
        key = file[:-5]  # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Prepare the request
            request = {"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body":
                {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ],
                "temperature": 0.002
            }}
            batch_requests.append(request)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    batch_files = []
    for i in range(0, len(batch_requests), batch_size):
        batch_file_path = os.path.join(output_dir, f"batchinput_{i//batch_size}.jsonl")
        with open(batch_file_path, "w") as f:
            for request in batch_requests[i:i + batch_size]:
                f.write(json.dumps(request) + "\n")
        batch_files.append(batch_file_path)
    return batch_files


def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if args.num_chunks > 1:
        pred_contents = []
        for _idx in range(args.num_chunks):
            file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.json")
            pred_contents += [json.loads(line) for line in open(file)]
        
    else:
        file = os.path.join(args.pred_path, f"pred.json")
        pred_contents = [json.loads(line) for line in open(file)]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['id'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred, "a_type": sample['answer_type'] if 'answer_type' in sample else None}
        prediction_set[id] = qa_set

    batch_files = prepare_batch_file(prediction_set, caption_files, output_dir, args.batch_size)


    # Upload the batch file
    client = OpenAI(api_key=args.api_key)
    batch_ids = []
    batch_input_file_ids = []
    for batch_file_path in batch_files:
        batch_input_file = client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )

        # Create the batch
        batch_input_file_id = batch_input_file.id
        batch_input_file_ids.append(batch_input_file_id)

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "eval job"}
        )

        print(batch)
        batch_id = batch.id
        batch_ids.append(batch_id)
        sleep(2)

    with open(os.path.join(output_dir, "batch_ids.txt"), "w") as f:
        for batch_id in batch_ids:
            f.write(batch_id + "\n")
    with open(os.path.join(output_dir, "batch_input_file_ids.txt"), "w") as f:
        for batch_input_file_id in batch_input_file_ids:
            f.write(batch_input_file_id + "\n")
    #
    # # Combine all the processed files into one
    # combined_contents = {}
    # json_path = args.output_json
    #
    # # Iterate through json files
    # for file_name in os.listdir(output_dir):
    #     if file_name.endswith(".json"):
    #         file_path = os.path.join(output_dir, file_name)
    #         with open(file_path, "r") as json_file:
    #             content = json.load(json_file)
    #             assert 'pred' in content[0], f"Error: {file_name} don't has key=pred"
    #             assert 'score' in content[0], f"Error: {file_name} don't has key=score"
    #             combined_contents[file_name[:-5]] = content
    #
    # # Write combined content to a json file
    # with open(json_path, "w") as json_file:
    #     json.dump(combined_contents, json_file)
    # print("All evaluation completed!")
    #
    # class ScoreMeter:
    #     def __init__(self):
    #         self.score_sum = 0
    #         self.count = 0
    #         self.yes_count = 0
    #         self.no_count = 0
    #         self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}
    #
    #     def add_score(self, score, pred):
    #         self.score_sum += score
    #         self.count += 1
    #         pred_lower = pred.lower()
    #         if 'yes' in pred_lower:
    #             self.yes_count += 1
    #             self.score_dict['yes'][score] += 1
    #         elif 'no' in pred_lower:
    #             self.no_count += 1
    #             self.score_dict['no'][score] += 1
    #
    #     def get_average_score(self):
    #         res = (self.score_sum / self.count) if self.count else 0
    #         return f"{res:.6f}"
    #
    #     def get_accuracy(self, response_type):
    #         if response_type == 'yes':
    #             res =  (self.yes_count / self.count) if self.count else 0
    #         elif response_type == 'no':
    #             res = (self.no_count / self.count) if self.count else 0
    #         else:
    #             res = 0
    #         return f"{res:.6f}"
    #
    # meter_dic = {'total': ScoreMeter()}
    # for key, result in combined_contents.items():
    #     # Computing score
    #     score_match = result[0]['score']
    #     score = int(score_match)
    #     pred = result[0]['pred']
    #
    #     meter_dic["total"].add_score(score, pred)
    #     if 'a_type' in result[1] and result[1]['a_type'] is not None:
    #         typ = str(result[1]['a_type'])
    #         if typ not in meter_dic:
    #             meter_dic[typ] = ScoreMeter()
    #         meter_dic[typ].add_score(score, pred)
    #
    #         if 'next' in args.output_dir:
    #             typ = typ[0]
    #             if typ not in meter_dic:
    #                 meter_dic[typ] = ScoreMeter()
    #             meter_dic[typ].add_score(score, pred)
    #
    # csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}
    #
    # output = ""
    # output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    # output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    # output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    # output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    # output += "\n"
    # output += "Total Score Yes/No distribution:\n"
    # for key, value in meter_dic["total"].score_dict.items():
    #     output += f"{key}:\n"
    #     for k in range(0, 6):
    #         v = value[k]
    #         output += f"{k}: {v}\n"
    # output += "\n"
    # output += "Answer Type Score distribution:\n"
    # output += 'Type, Accuracy, Avg_score\n'
    # key_list = sorted([k for k in meter_dic.keys()])
    # for key in key_list:
    #     output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
    #     csv_dic[key] = meter_dic[key].get_accuracy('yes')
    #
    # output += "\n"
    # for k in csv_dic.keys():
    #     output += f"{k}, "
    # output = output.rstrip(', ')  # Remove the trailing comma and space
    # output += "\n"
    #
    # for k in csv_dic.keys():
    #     output += str(csv_dic[k]) + ", "
    # output = output.rstrip(', ')  # Remove the trailing comma and space
    # output += "\n"
    #
    # print(output)
    # args.output_csv = args.output_json.replace(".json", ".csv")
    # with open(args.output_csv, 'w') as f:
    #     f.write(output)

if __name__ == "__main__":
    main()

