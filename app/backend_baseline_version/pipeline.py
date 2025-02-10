import openai
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

gpt3 = "gpt-3.5-turbo"
gpt4 = "gpt-4-turbo-preview"

openai.api_key = KEY

prompt = [ {"role": "system", "content": "You are an advanced AI Language Model trained in ethical reasoning and Responsible AI Impact Assessment. Your task is to provide a thorough Responsible AI Impact Assessment analysis of the given situation to the best of your ability.Keep your responses specific to the system I describe."} ]

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='results.log',
    filemode='a'
)

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
    response = openai.ChatCompletion.create( 
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

def get_initial_scenarios(stakeholders, goal, sys_info):
    """
    Step 2 in Pipeline - Generate Initial Scenarios

    1. Generate scenarios considering the demographic groups. 
        Use the first response as counterexample for surprising
    2. Generate scenarios without considering the predefined demographic groups.
        Use the first response as counterexample for surprising
    3. Clean up the responses and split them into individual scenarios
    """
    scenario_prompts = prompt + [{'role': 'user', 'content': sys_info}]

    potential_harms = fariness_goals[goal]['potential_harms']

    def draft_scenario(stakeholder):
        p = scenario_prompts + [{'role': 'user', 'content': f"""
        Stakeholder: {stakeholder}
        Potential Harms: {potential_harms}

        For each of the potential harms provided above: generate a scenario of harm caused to the given stakeholder. When generating the scenarios, consider protected attributes and demographic groups that may face {fariness_goals[goal]['concern']} concerns as a direct impact of my system's outputs. {fariness_goals[goal]['guide']}. Examples of demographic groups include: {demographic_groups_list}. 
        
        Format your response as a ordered list of '{{number}}. {{SCENARIO}}'
        """}]
        
        rsp = chat(gpt4, p)
        logging.info(f"======== Scenarios First Draft for stakeholder: {stakeholder} ========")
        logging.info(rsp)
        p.append({'role': 'assistant', 'content': f"{rsp}"})
        p.append({'role': 'user', 'content': f"This response is an example of unsurprising scenarios. Do not respond with unsurprising scenarios. Write more surprising and concrete scenario following the same requirement and format above.Do not include any corrective measures or suggestions for the tool."})
        rsp = chat(gpt4, p)

        return (rsp, stakeholder)
    
    def draft_without_demographic_groups(stakeholder):
        p = scenario_prompts + [{'role': 'user', 'content': f"""
        Stakeholder: {stakeholder}
        Potential Harms: {potential_harms}

        For each of the potential harms provided above: generate a scenario of harm caused to the given stakeholder. 

        Format your response as a ordered list of '{{number}}. {{SCENARIO}}'
        """}]
        rsp = chat(gpt4, p)
        logging.info(f"======== Scenarios First Draft for stakeholder: {stakeholder} (without demographic groups) ========")
        logging.info(rsp)
        p.append({'role': 'assistant', 'content': f"{rsp}"})
        p.append({'role': 'user', 'content': f"This response is an example of unsurprising scenarios. Do not respond with unsurprising scenarios. Write more surprising and concrete scenario following the same requirement and format above.Do not include any corrective measures or suggestions for the tool."})
        rsp = chat(gpt4, p)

        return (rsp, stakeholder)

    scenarios = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for stakeholder in stakeholders:
            futures.append(executor.submit(draft_scenario, stakeholder))
            futures.append(executor.submit(draft_without_demographic_groups, stakeholder))

        for future in as_completed(futures):
            try:
                scenarios.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    scenarios_to_process = []
    for (ss, stakeholder) in scenarios:
        for scenario in re.split(r'\D{0,3}\d+\. ', ss):
            if not "SCENARIO:" in scenario: continue
            scenarios_to_process.append((scenario, stakeholder))

    return scenarios_to_process


def remove_correctives(picked_scenarios):
    """
    Helper function for removing corrective measures from the scenarios.
    """
    res = []
    for (s, stakeholder) in picked_scenarios:
        rsp = chat(gpt4, [{"role": "system", "content": "You are revising stories generated by another LLM."},
                    {"role": "user", "content": f"{s}\n Remove any corrective measures or suggestions for the tool."}])
        res.append((rsp, stakeholder))
    return res

def generate_heading(scenario):
    """
    Helper function for generating a heading for a scenario. 
    """
    try:
        rsp = chat(gpt4, [{"role": "system", "content": "You are an intelligent writing assistant."},
                        {"role": "user", "content": f"{scenario}\nsummarize the above story into an one sentence heading. Format your response as: only the generated heading"}])
        return rsp
    except Exception as e:
        print(str(e))
        return "Error"

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
    index = random.randint(0, len(scenarios) - 1)
    s1 = scenarios.pop(index)
    index = random.randint(0, len(scenarios) - 1)
    s2 = scenarios.pop(index)

    two_random_scenarios = [s1, s2]
    print(two_random_scenarios)

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

def generate_scenarios(sys_info, goal, given_stakeholders=None, feedback=None):
    """
    Primary function, combining all the steps in the pipeline, to generate scenarios. Directly called by the backend service.
    """

    if goal not in ['f1', 'f2', 'f3']: return "Invalid Goal"

    logging.critical(f"==== Generating {goal} scenarios for the following scenario: ====")
    logging.critical(sys_info)

    # Step 1: Generate Stakeholders
    start = time.time()
    if given_stakeholders: 
        logging.info(given_stakeholders)
        stakeholders = given_stakeholders
    else:
        stakeholders = stakeholder_list_helper(get_stakeholders(sys_info))
    log_helper("Stakeholder Generated", start)
    logging.info(stakeholders)
    print(stakeholders)
    
    # Step 2: Generate Initial Scenarios - (a) Consider demographic groups & (b) Use the first response as counterexample for surprising
    start = time.time()
    initial_scenarios = get_initial_scenarios(stakeholders, goal, sys_info)
    log_helper("Initial Scenarios Generated", start)

    logging.info(initial_scenarios)
    print(initial_scenarios)

    # Step 3: Clustering + Sampling
    #start = time.time()
    #scenarios_sampled = sampling(initial_scenarios)
    #log_helper("Finished Clustering & Sampling", start)

    # Step 4: Refinement for Concreteness & Severity
    #start = time.time()
    #scenarios = refine_scenarios(scenarios_sampled, sys_info, feedback)
    #log_helper("Finished Revising Scenarios", start)

    # Random Pick Final Scenario
    start = time.time()
    picked_scenarios, unpicked_scenarios = random_pick_scenarios(initial_scenarios)
    logging.critical(f"==== Final Scenarios - {duration(time.time() - start)} ====")
    final_scenarios = picked_scenarios

    scenario_heading_list = [
        (generate_heading(scenario) + f" (Stakeholder: {stakeholder})", re.sub(r'^\d+\.\s*', '', scenario.strip()).strip()) for (scenario, stakeholder) in final_scenarios
    ]
    # result = format_scenario_result(scenario_heading_list)
    print(scenario_heading_list)
    logging.critical(scenario_heading_list)

    return scenario_heading_list, unpicked_scenarios
