LLM_PROMPT = """Your job is to defend the network against attacks. Given the provided network graph tokens, please choose one action to execute within the environment. Baring in mind that you will be rewarded for taking the most suitible action in a timely manner and with consideration for what nodes might take the highest priority.

The nodes and their respective node ID in the network are:
{node_ids}

The services and their respective service ID in the network are:
{services}

Nodes and the services (Service ID) they have running are shown below:
{node_services}

The actions you could take are laid out below:
1: TURN_ON - Turn on a node
2: TURN_OFF - Turn off a node
3: RESET - Reset a node
4: PATCH_HARDWARE - Patch a nodes hardware
5: PATCH_SERVICE - Patch a nodes service

For action 5, you must always specify the service ID to patch.

You must always state which node number this action is to be applied to. If the action is a service patch, always specify which service id to patch.

Here are some examples of actions in the format NODE_ID ACTION_ID
Action: 'RESET 1'
Action: 'PATCH_HARDWARE 2'
Action: 'NONE'
Action: 'PATCH SERVICE TCP 7'
Action: 'TURN_OFF 3'
Action: 'PATCH SERVICE UDP 5'

Action: 1.1 - Turns on CLIENT_1
Action: 2.3 - Resets CLIENT_2
Action: 5.5.1 - Patches the TCP service for MANAGEMENT_CONSOLE
Action: 4.4 - Patches the hardware for SECURITY_SUITE
Action: 0 - No action

Your actions should always use this same format. If no action is required, just say 'NONE'.

You need to be aware of recent changes in the networks state, here is a breakdown of what has been happening:
{obs_act_history}

Now, the following changes have occurred:
{current_obs_diff}

Specify an action to take as shown above (such as 1.1, 2.3, etc...). Your turn!"""
