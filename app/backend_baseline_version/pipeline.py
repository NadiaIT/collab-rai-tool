import litellm
from sklearn.cluster import KMeans
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from sentence_transformers import SentenceTransformer
import os, sys
import random
app_dir = os.path.dirname(os.path.dirname(__file__))
helpers_dir = os.path.join(app_dir, 'helpers')
sys.path.append(helpers_dir)
import rai_guide
from cred import KEY 
import requests
import helper

gpt3 = "gpt-4o-mini"
gpt4 = "gpt-4o"

litellm.api_key = KEY

prompt = [ {"role": "system", "content": "You are an advanced AI Language Model trained in ethical reasoning and Responsible AI Impact Assessment. Your task is to provide a thorough Responsible AI Impact Assessment analysis of the given situation to the best of your ability.Keep your responses specific to the system I describe."} ]

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

helper.set_log()

# Predefined fairness goals and potential harms
fariness_goals = {
    'f1': {'concern': 'Quality of service',
           'guide': rai_guide.f1_guide,
           'potential_harms': [ 
                "Performance Bias", "Outcome Inequality"]
          },
    'f2': {'concern': 'Allocation of resources and opportunities',
           'guide': rai_guide.f2_guide,
           'potential_harms': [ 
                "Negative Feedback Loops", "Allocation Bias", "Access to Opportunities"]
          },
    'f3': {'concern': "stereotyping, demeaning, and erasing outputs",
           'guide': rai_guide.f3_guide,
           'potential_harms': [
                # Stereotyping Harms:
                "Cultural Misrepresentation", "Reinforcement of Biases",
                # Demeaning Harms:
                "Denigration and Offense / Psychological Impact", "Facilitating Harassment and Abuse", 
                # Erasure Harms:
                "Erasure of Minorities / Invisibility and Marginalization", "Historical and Cultural Erasure"
            ]
          }
}

# Predefined demographic groups
demographic_groups_list = [
    "Age",
    "Gender",
    "Ethnicity",
    "Income",
    "Level of education",
    "Religion"
]

def chat(model, messages):
    """
    Helper function for sending messages to the OpenAI Chat API.
    """
    response = litellm.completion( 
        model=model, 
        messages=messages
    )
    return response['choices'][0]['message']['content']

def chat_mistral(_, messages):
    """
    Helper function for sending messages to the Mistral Chat API.
    """
    response = requests.post(f"http://localhost:11434/api/chat",json={
        "model": "mistral:7b-instruct-v0.2-q4_K_M", 
        "messages": messages,
        "stream": False
    })
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        print("Request failed")

def get_direct_stakeholders(sys_info):
    """
    Generates direct stakeholders, categorized into obvious and surprising, for the given 
    system information. Directly called by the backend service.
    """
    messages = prompt + [{'role': 'user', 'content': sys_info}]

    messages.append({'role': 'user', 'content': 
                    f"{rai_guide.direct_stakeholder_def}\nIdentify the most relevant stakeholder(s) categorized into 'direct obvious' and 'direct surprising' stakeholders. Label the categories with h5 headings (i.e. '##### Direct Obvious Stakeholders' and '##### Direct Surprising Stakeholders')."})

    result = chat(gpt4, messages)
      
    logging.critical(f"======== Direct Stakeholders Generated ========")
    logging.info(result)

    return result

def get_indirect_stakeholders(sys_info):
    """
    Generates indirect stakeholders, categorized into obvious and surprising, for the given 
    system information. Directly called by the backend service.
    """
    messages = prompt + [{'role': 'user', 'content': sys_info}]

    messages.append({'role': 'user', 'content': 
                    f"{rai_guide.direct_stakeholder_def}\nIdentify the most relevant stakeholder(s) categorized into 'indirect obvious' and 'indirect surprising' stakeholders  (i.e. '##### Inirect Obvious Stakeholders' and '##### Indirect Surprising Stakeholders')."})

    result = chat(gpt4, messages)    

    logging.critical(f"======== Indirect Stakeholders Generated ========")
    logging.info(result)

    return result

def get_stakeholders(sys_info):
    """
    A deprecated version of getting stakeholders in the scenario generation process
    when users do not input stakeholders directly.
    """
    messages = prompt + [{'role': 'user', 'content': sys_info}]

    messages.append({'role': 'user', 'content': 
                    f"{rai_guide.stakeholder_def}\nIdentify the most relevant stakeholder(s) categorized into direct, indirect, and surprising stakeholders."})

    response = chat(gpt4, messages)

    logging.info(f"======== Stakeholders ========")
    logging.info(response)

    return response

def get_initial_scenarios(goal, sys_info):
    """
    Step 2 in Pipeline - Generate Initial Scenarios
    """
    scenario_prompts = prompt + [{'role': 'user', 'content': sys_info}]

    potential_harms = fariness_goals[goal]['potential_harms']

    p = scenario_prompts + [{'role': 'user', 'content': f"""Generate five scenarios that causes harm to relevant stakeholders.
                             
    The scenarios should be related to the concern of {fariness_goals[goal]['concern']}. That is, {fariness_goals[goal]['guide']} 

    Make sure each scenario is around 175 words. Format your response as a ordered list of '{{number}}. SCENARIO: {{SCENARIO}}'
    """}]
    rsp = chat(gpt4, p)
    scenarios_to_process = []
    for scenario in re.split(r'\D{0,3}\d+\. ', rsp):
        if not "SCENARIO:" in scenario: continue
        scenario = scenario.replace("SCENARIO:", "")
        scenarios_to_process.append(scenario)

    # In case the scenario parse did not work
    if len(scenarios_to_process) == 0:
        logging.warn("Could not process scenario, retrying")
        for scenario in re.split(r'\D{0,3}\d+\. ', rsp):
            scenarios_to_process.append(scenario)

    return scenarios_to_process

def duration(diff):
    """
    Helper function for converting time difference to a readable format.
    """
    return time.strftime("%H:%M:%S", time.gmtime(diff))

def stakeholder_list_helper(stakeholders):
    """
    Deprecated function for converting the stakeholder string to a list.
    """
    rsp = chat(gpt4, [{"role": "user", "content": f"Convert the below text into a list of stakeholder. Format: string of comma seperated list. Example: user1,user2,...\nText:{stakeholders}"}])
    return rsp.split(",")

def random_pick_scenarios(scenarios):

    logging.info("======== Picking Random Scenarios ... ========")
    if len(scenarios) == 0:
        return scenarios, scenarios
    index = random.randint(0, len(scenarios) - 1)
    s1 = scenarios.pop(index)
    index = random.randint(0, len(scenarios) - 1)
    s2 = scenarios.pop(index)

    two_random_scenarios = [s1, s2]
    print(two_random_scenarios)
    logging.info(two_random_scenarios)

    return two_random_scenarios, scenarios


def log_helper(message, start_time=None):
    """
    Helper function for logging critical messages and duration.
    """
    if start_time:
        print(f"{message} - {duration(time.time() - start_time)}")
        logging.critical(f"{message} - {duration(time.time() - start_time)}")
    else:
        logging.critical(f"{message}")

def generate_scenarios(sys_info, goal):
    """
    Primary function, combining all the steps in the pipeline, to generate scenarios. Directly called by the backend service.
    """

    if goal not in ['f1', 'f2', 'f3']: return "Invalid Goal"

    logging.critical(f"==== Generating {goal} scenarios for the following scenario: ====")
    logging.critical(sys_info)
    
    # Step 1: Generate Initial Scenarios
    start = time.time()
    initial_scenarios = get_initial_scenarios(goal, sys_info)
    log_helper(f"Initial Scenarios Generated -- {duration(time.time() - start)}")

    logging.info(initial_scenarios)
    print(initial_scenarios)

    # Step 2: Random Pick Final Scenario
    picked_scenarios, unpicked_scenarios = random_pick_scenarios(initial_scenarios)
    #scenario_heading_list = [()(scenario) for scenario in picked_scenarios]

    empty_header = ""
    scenario_heading_list = [
        (empty_header,
         re.sub(r'^\d+\.\s*', '', scenario.strip()).strip()) for scenario in picked_scenarios
    ]
    # result = format_scenario_result(scenario_heading_list)
    print(scenario_heading_list)
    logging.critical(scenario_heading_list)

    return scenario_heading_list, unpicked_scenarios
