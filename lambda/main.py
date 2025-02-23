import os
from datetime import date
import requests
import json
import boto3

# Env variables
MODEL_TYPE = os.environ['model_type']
NYT_API = os.environ['nyt_api']
NYT_API_KEY = os.environ['nyt_api_key']
NYT_QUERY = os.environ['nyt_query']
NYT_NEWS_DESK = os.environ['nyt_news_desk']
NYT_TYPE_OF_MATERIAL = os.environ['nyt_type_of_material']
PROMPT = os.environ['prompt']
APP_ID = os.environ['app_id']
BRANCH = os.environ['branch']
S3_BUCKET = os.environ['s3_bucket']
S3_KEY_DATA_FILE = os.environ['s3_key_data_file']
DEPLOY_ONLY = os.environ['deploy_only']

s3 = boto3.resource('s3')
    
bedrock = boto3.client(
    service_name='bedrock', 
    region_name='us-east-1'
)
    
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
)

def call_nyt():
    try:
        current_date = date.today().strftime("%Y%m%d")
        query_params = {
            "api-key": NYT_API_KEY,
            "begin_date": current_date,
            "end_date": current_date,
            "q": NYT_QUERY,
            "news_desk": NYT_NEWS_DESK,
            "type_of_material": NYT_TYPE_OF_MATERIAL
        }
        response = requests.get(NYT_API, params=query_params)
        response.raise_for_status()
        articles = []

        for item in response.json()["response"]["docs"]:
            article = {"url": item["web_url"], "abstract": item["abstract"] + item["lead_paragraph"]}
            articles.append(article)           

        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error making GET request: {e}")
        return None

def call_mistral(articles):
    article_text = ""
    for article in articles:
        article_text += article["abstract"] + " " + article["url"]

    body = json.dumps(
        {
            "prompt": "<s>[INST]" + PROMPT + article_text + "[/INST]", 
            "max_tokens": 1000,
            "temperature": 0.9,
            "top_p": 0.9,
        }
    )
     
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId='mistral.mixtral-8x7b-instruct-v0:1', 
        accept='application/json', 
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('outputs')
    return answer

def call_claude(articles):
    article_text = ""
    for article in articles:
        article_text += article["abstract"] + " " + article["url"]

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.9,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": PROMPT + article_text
                        }
                    ],
                }
            ]
        }
    )
    
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId='arn:aws:bedrock:us-east-1:959425594836:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0', 
        accept='application/json', 
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    answer = response_body["content"]
    return answer

def read_from_s3():
    content_object = s3.Object(S3_BUCKET, S3_KEY_DATA_FILE)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    return json.loads(file_content)

def write_to_s3(new_data):
    content_object = s3.Object(S3_BUCKET, S3_KEY_DATA_FILE)
    content_object.put(Body=json.dumps(new_data))

def add_to_data(current_data, new_data):
    for element in current_data:
        new_data.append(element)
    write_to_s3(new_data)

def lambda_handler(event, context):
    try:
        data = read_from_s3()
        nyt_response = call_nyt()
        model_response = None

        if MODEL_TYPE == 'claude':
            model_response = call_claude(nyt_response)
        elif MODEL_TYPE == 'mistral':
            model_response = call_mistral(nyt_response)

        add_to_data(data, json.loads(model_response[0]['text']))

        print("Job done.")
    except:
        raise