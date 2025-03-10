import os
from datetime import date
from urllib.request import Request, urlopen
import requests
import json
import boto3

s3 = boto3.resource('s3')
    
# Bedrock client used to interact with APIs around models
bedrock = boto3.client(
    service_name='bedrock', 
    region_name='us-east-1'
)
    
# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', 
    region_name='us-east-1'
)

MODEL_TYPE = os.environ['model_type']
NYT_API = os.environ['nyt_api']
NYT_API_KEY = os.environ['nyt_api_key']
NYT_QUERY = os.environ['nyt_query']
NYT_FQ = os.environ['nyt_fq']
APP_ID = os.environ['app_id']
BRANCH = os.environ['branch']
S3_BUCKET = os.environ['s3_bucket']
S3_KEY_DATA_FILE = os.environ['s3_key_data_file']

PROMPT = "You are an unbiased political analyst. You carefully analyse news articles every day as part of chronicling President Donald J. Trump's 2nd term. Below is a collection of articles from today. Summarise each article into a single succinct sentence highlighting actions and activities by Donald J. Trump and his administration, including Elon Musk's DOGE, that are ethically/morally questionable, potentially illegal, and that undermine American institutions and the American constitution or would otherwise be considered \"dumb\" by any sensible or reasonable individual. It is vital you are as objective as possible and do not embellish or read into details that are not there. Only include articles that are directly related to President Trump, his administration, Elon Musk or DOGE. If there is nothing questionable or improper, then ignore the article. Where articles describe the same thing, combine them into a single summary. Phrase summaries in the third person. Cite the respective article URL or set of URLs if articles are combined along with the summary. Provide an assessment of severity (LOW, MED, HIGH) of the implications of the actions or activities reported in the articles on American democracy, the global order, and the rule of law. Identify a list of key one word \"tags\" for each entry. Leave out the preamble and postamble. Return the summaries in JSON format with the following schema: [ { \"summary\" : <summary>, \"urls\" : [<urls>], \"severity\" : <severity>, \"tags\" : [<tags>] } ]"

def call_nyt(search_date):
    try:
        current_date = search_date 
        query_params = {"api-key": NYT_API_KEY,
                   "begin_date": current_date,
                   "end_date": current_date,
                   "q": NYT_QUERY,
                   "fq": NYT_FQ}
        response = requests.get(NYT_API, params=query_params)
        response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)
        articles = []

        for item in response.json()["response"]["docs"]:
            #print(json.dumps(item) + "\n")
            article = {
                "url": item["web_url"], 
                "abstract": item["headline"]["main"] + item["abstract"] + item["lead_paragraph"]
            }
            #print(article)
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
                    "content": [{"type": "text", "text": PROMPT + article_text}],
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
    content_object.put(Body=json.dumps(new_data), ACL='public-read')

def add_to_data(current_data, new_data):
    for element in current_data:
        element['featured'] = "false"
        new_data.append(element)
    write_to_s3(new_data)

def lambda_handler(event, context):

    search_date = date.today().strftime("%Y%m%d")

    if ("search_date" in event):
        search_date = event['search_date']

    data = read_from_s3()
    nyt_response = call_nyt(search_date)
    model_response = None

    if MODEL_TYPE == 'claude':
        model_response = call_claude(nyt_response)
    elif MODEL_TYPE == 'mistral':
        model_response = call_mistral(nyt_response)

    print(model_response)
    if (len(model_response) > 0):
        new_data = json.loads(model_response[0]['text'])
        for element in new_data:
            element['date'] = search_date
            element['featured'] = "true"

        #print(new_data)

        add_to_data(data, new_data)

    try:
        print("Job done.")
        #if model_response is None:
        #    raise ValueError('model_response empty')
    except:
        raise
    else:
        return event['time']
    finally:
        print("Run complete.")