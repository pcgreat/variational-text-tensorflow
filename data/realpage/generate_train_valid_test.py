import pdb
import random

import pandas as pd

df = pd.read_csv("wayblazer_chattrans_2018_cln.csv", sep="|", error_bad_lines=False,
                 warn_bad_lines=True,
                 low_memory=False)
df = df[df.event_type == 'chat']

training_users = []
training_agents = []
for _, _df in df.groupby("channelSid"):
    i = 0

    user_texts = []
    agent_texts = []
    while i < len(_df):
        if _df.iloc[i]["isAgent"]:  # if is agent
            if user_texts and pd.notnull(_df.iloc[i]["content"]):
                agent_texts.append(_df.iloc[i]["content"].strip())
        else:
            if user_texts and agent_texts:
                training_users.append(" ".join(user_texts))
                training_agents.append(" ".join(agent_texts))
                user_texts = []
                agent_texts = []
            elif pd.notnull(_df.iloc[i]["content"]):
                user_texts.append(_df.iloc[i]["content"].strip())
        i += 1
    if user_texts and agent_texts:
        training_users.append(" ".join(user_texts))
        training_agents.append(" ".join(agent_texts))
        user_texts = []
        agent_texts = []

with open("train_usr.txt", "w") as f:
    with open("train_agent.txt", "w") as g:
        for usr_txt, agt_txt in zip(training_users, training_agents):
            f.write(usr_txt + "\n")
            g.write(agt_txt + "\n")

pdb.set_trace()


trains = []
valids = []
tests = []
for line in lines:
    p = random.random()
    if p < 0.8:
        trains.append(line)
    elif p < 0.9:
        valids.append(line)
    else:
        tests.append(line)

with open("train.txt", "w") as f:
    f.write("".join(trains))
with open("valid.txt", "w") as f:
    f.write("".join(valids))
with open("test.txt", "w") as f:
    f.write("".join(tests))
