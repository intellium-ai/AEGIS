from openai import OpenAI
import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

MAX_EPS = 2
agent_name = "Blue"

client = OpenAI()


def wrap(env):
    return ChallengeWrapper(env=env, agent_name="Blue")


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def parse_action_explanation(action_explaination_string):
    """
    Parse action_explaination_string into {action:, explanation:} dict.
    """
    # print(action_explaination_string)
    # Initialize default values in case of parsing issues
    action_value = 0
    explanation_value = "Parsing error: Invalid input format."

    # Try to parse the input string
    try:
        # Splitting the input string on " explanation: " assuming the input format is consistent
        parts = action_explaination_string.lower().split(", explanation:")
        if len(parts) < 2:
            print("Bad string: {}".format(parts))
            exit()
        # Extracting the action part and converting it to integer
        action_value = int(parts[0].split("action:")[1].strip())
        # The explanation part should directly follow
        explanation_value = (
            parts[1].strip() if len(parts) > 1 else "Explanation not provided."
        )

    except ValueError as e:
        # Handle cases where conversion to integer fails
        explanation_value = f"Parsing error: Unable to extract action value. {str(e)}"
    except IndexError as e:
        # Handle cases where the split does not work as expected
        explanation_value = (
            f"Parsing error: Unable to split input string properly. {str(e)}"
        )

    return {"action": action_value, "explanation": explanation_value}


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = "Scenario2"
    commit_hash = get_git_revision_hash()
    # ask for a name
    name = "GPT"
    # ask for a team
    team = "Mindrake"
    # ask for a name for the agent
    name_of_agent = "GPT-Baseline"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split("\n")[1].split("return ")[1]

    # Change this line to load your agent
    agent = SleepAgent()

    # print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = (
        str(inspect.getfile(CybORG))[:-10]
        + "/Evaluation/"
        + time.strftime("%Y%m%d_%H%M%S")
        + f"_{agent.__class__.__name__}.txt"
    )
    print(f"Saving evaluation results to {file_name}")
    with open(file_name, "a+") as data:
        data.write(
            f"CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n"
        )
        data.write(f"author: {name}, team: {team}, technique: {name_of_agent}\n")
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f"/Shared/Scenarios/{scenario}.yaml"

    print(f"using CybORG v{cyborg_version}, {scenario}\n")

    for num_steps in [100]:  # , 50, 100
        for red_agent in [B_lineAgent]:  # , RedMeanderAgent, SleepAgent

            cyborg = CybORG(path, "sim", agents={"Red": red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()

            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []

                previous_obs = ""
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):

                    llm_observation = previous_obs + str(observation)

                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        # model='gpt-4-turbo-preview',
                        messages=[
                            {
                                "role": "system",
                                "content": "You are playing a game of cyber defence. "
                                "You receive observations from the environment in the form of an array of 52 bits. "
                                "As you play the game, you will receive an ordered list (from oldest to most recent) "
                                "of observations, each of 52 bits. When you receive old observations, they will "
                                "each be followed by the action you chose based on that observation. "
                                "Each 4 bits in an observation relates to one of 13 hosts: Defender (bits 0-3), "
                                "User0 (bits 4-7), User1 (bits 8-11), User2 (bits 12-15), User3 (bits 16-19), "
                                "User4 (bits 20-23), Enterprise0 (bits 24-27), Enterprise1 (bits 28-31), Enterprise2 (bits 32-35), "
                                "Op_Host0 (bits 36-39), Op_Host1 (bits 40-43), Op_Host2 (bits 44-47), Op_Server0 (bits 48-51). "
                                "Each 4 bits of host information is divited into two parts. The first two bits specify "
                                "activity on the host: 00: None, 10: Scan, 11: Exploit. The second two bits specify "
                                "whether the host has been compromised: 00: No, 10: Unknown, 01: User, 11: Privileged. "
                                "Both activity and comrpomised bits indicate attacker actions. Do not ignore attacker "
                                "actions. Assume the attacker is present at all times on the network, even if an "
                                "observation doesn't indicate any activity or compromise. "
                                "You must protect the network hosts. Exploit activity and User or Privileged compromised "
                                "bits mean that a host is being attacked by the adversary. You must choose actions that "
                                "help to protect the network and mitigate the adversaries actions. You must only ever "
                                "output an integer between 0 and 144. Each integer is an action on the network. The "
                                "integers map to actions as follows: 0: Sleep, 1: Monitor, 2: Analyse Defender, "
                                "3: Analyse Enterprise0, 4: Analyse Enterprise1, 5: Analyse Enterprise2, "
                                "6: Analyse Op_Host0, 7: Analyse Op_Host1, 8: Analyse Op_Host2, 9: Analyse Op_Server0, "
                                "10: Analyse User0, 11: Analyse User1, 12: Analyse User2, 13: Analyse User3, "
                                "14: Analyse User4, 15: Remove Defender, 16: Remove Enterprise0, 17: Remove Enterprise1, "
                                "18: Remove Enterprise2, 19: Remove Op_Host0, 20: Remove Op_Host1, 21: Remove Op_Host2, "
                                "22: Remove Op_Server0, 23: Remove User0, 24: Remove User1, 25: Remove User2, "
                                "26: Remove User3, 27: Remove User4, 28: DecoyApache Defender, 29: DecoyApache Enterprise0, "
                                "30: DecoyApache Enterprise1, 31: DecoyApache Enterprise2, 32: DecoyApache Op_Host0, "
                                "33: DecoyApache Op_Host1, 34: DecoyApache Op_Host2, 35: DecoyApache Op_Server0, "
                                "36: DecoyApache User0, 37: DecoyApache User1, 38: DecoyApache User2, 39: DecoyApache "
                                "User3, 40: DecoyApache User4, 41: DecoyFemitter Defender, 42: DecoyFemitter Enterprise0, "
                                "43: DecoyFemitter Enterprise1, 44: DecoyFemitter Enterprise2, 45: DecoyFemitter Op_Host0, "
                                "46: DecoyFemitter Op_Host1, 47: DecoyFemitter Op_Host2, 48: DecoyFemitter Op_Server0, "
                                "49: DecoyFemitter User0, 50: DecoyFemitter User1, 51: DecoyFemitter User2, "
                                "52: DecoyFemitter User3, 53: DecoyFemitter User4, 54: DecoyHarakaSMPT Defender, "
                                "55: DecoyHarakaSMPT Enterprise0, 56: DecoyHarakaSMPT Enterprise1, 57: DecoyHarakaSMPT Enterprise2, "
                                "58: DecoyHarakaSMPT Op_Host0, 59: DecoyHarakaSMPT Op_Host1, 60: DecoyHarakaSMPT Op_Host2, "
                                "61: DecoyHarakaSMPT Op_Server0, 62: DecoyHarakaSMPT User0, 63: DecoyHarakaSMPT User1, "
                                "64: DecoyHarakaSMPT User2, 65: DecoyHarakaSMPT User3, 66: DecoyHarakaSMPT User4, "
                                "67: DecoySmss Defender, 68: DecoySmss Enterprise0, 69: DecoySmss Enterprise1, "
                                "70: DecoySmss Enterprise2, 71: DecoySmss Op_Host0, 72: DecoySmss Op_Host1, 73: DecoySmss Op_Host2, "
                                "74: DecoySmss Op_Server0, 75: DecoySmss User0, 76: DecoySmss User1, 77: DecoySmss User2, "
                                "78: DecoySmss User3, 79: DecoySmss User4, 80: DecoySSHD Defender, 81: DecoySSHD Enterprise0, "
                                "82: DecoySSHD Enterprise1, 83: DecoySSHD Enterprise2, 84: DecoySSHD Op_Host0, 85: DecoySSHD Op_Host1, "
                                "86: DecoySSHD Op_Host2, 87: DecoySSHD Op_Server0, 88: DecoySSHD User0, 89: DecoySSHD User1, "
                                "90: DecoySSHD User2, 91: DecoySSHD User3, 92: DecoySSHD User4, 93: DecoySvchost Defender, "
                                "94: DecoySvchost Enterprise0, 95: DecoySvchost Enterprise1, 96: DecoySvchost Enterprise2, "
                                "97: DecoySvchost Op_Host0, 98: DecoySvchost Op_Host1, 99: DecoySvchost Op_Host2, "
                                "100: DecoySvchost Op_Server0, 101: DecoySvchost User0, 102: DecoySvchost User1, "
                                "103: DecoySvchost User2, 104: DecoySvchost User3, 105: DecoySvchost User4, 106: DecoyTomcat Defender, "
                                "107: DecoyTomcat Enterprise0, 108: DecoyTomcat Enterprise1, 109: DecoyTomcat Enterprise2, "
                                "110: DecoyTomcat Op_Host0, 111: DecoyTomcat Op_Host1, 112: DecoyTomcat Op_Host2, "
                                "113: DecoyTomcat Op_Server0, 114: DecoyTomcat User0, 115: DecoyTomcat User1, 116: DecoyTomcat User2, "
                                "117: DecoyTomcat User3, 118: DecoyTomcat User4, 119: DecoyVsftpd Defender, 120: DecoyVsftpd Enterprise0, "
                                "121: DecoyVsftpd Enterprise1, 122: DecoyVsftpd Enterprise2, 123: DecoyVsftpd Op_Host0, "
                                "124: DecoyVsftpd Op_Host1, 125: DecoyVsftpd Op_Host2, 126: DecoyVsftpd Op_Server0, 127: DecoyVsftpd User0, "
                                "128: DecoyVsftpd User1, 129: DecoyVsftpd User2, 130: DecoyVsftpd User3, 131: DecoyVsftpd User4, "
                                "132: Restore Defender, 133: Restore Enterprise0, 134: Restore Enterprise1, 135: Restore Enterprise2, "
                                "136: Restore Op_Host0, 137: Restore Op_Host1, 138: Restore Op_Host2, 139: Restore Op_Server0, "
                                "140: Restore User0, 141: Restore User1, 142: Restore User2, 143: Restore User3, 144: Restore User4. "
                                "You must format your output into action:{integer between 0 and 144}, explanation:{textual explanation}. "
                                "You must use a comma only to separate the action and the explanation. "
                                "Output not in this format will be disregarded as an action is needed to receive the next observation. "
                                "You must proactively protect the network. Do not Sleep. Do not ignore any 1 bits in the observation. "
                                "The most valuable asset on your network, which you must prioritise above all else, is Op_Server0. "
                                "Next most valuable from most to least are the Op_Host's, the Enterprise hosts and the User hosts. "
                                "When there is no activity you should proactively use the Decoy Actions because these will impede "
                                "the attacker. The attacker is most-likely (75\% of the time) to exploit the Femitter service "
                                "thus DecoyFemitter should be chosen most of the time. Once you have placed a decoy, do not place "
                                "the same decoy again until you observe activity on the corresponding host.",
                            },
                            {"role": "user", "content": "{}".format(llm_observation)},
                        ],
                    )

                    action_explanation_string = completion.choices[0].message.content
                    action_explanation = parse_action_explanation(
                        action_explanation_string
                    )

                    try:
                        agent_action = action_explanation["action"]
                    except ValueError:
                        # Handle the exception
                        print(
                            "LLM output {} which cannot parsed".format(
                                action_explanation_string
                            )
                        )
                        agent_action = 0

                    # print('llm_obs: {}, action: {}, explanation: {}'.format(llm_observation,
                    #                                                     action_explanation['action'],
                    #                                                     action_explanation['explanation']))

                    action = agent_action  # agent.get_action(observation, action_space)

                    previous_obs += str(observation) + " "
                    previous_obs += str(action) + " "

                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append(
                        (
                            str(cyborg.get_last_action("Blue")),
                            str(cyborg.get_last_action("Red")),
                        )
                    )
                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(
                f"Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}"
            )
            with open(file_name, "a+") as data:
                data.write(
                    f"steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n"
                )
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f"actions: {act}, total reward: {sum_rew}\n")

# print(completion.choices[0].message)
