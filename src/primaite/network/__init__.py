from primaite.network.network import Network

QUESTION_VARIATIONS = {
    "How many nodes are there?": [
        "What is the total number of nodes?",
        "Can you tell me the count of nodes in the network?",
        "How many total nodes are present?",
        "What’s the node count in the network?",
        "Could you specify the number of nodes?"
    ],
    "How many links are there in the network?": [
        "What is the total number of links in the network?",
        "How many links can be found in this network?",
        "Can you tell me the number of links in the network?",
        "What’s the total count of links in the network?",
        "How many links are there overall?"
    ],
    "How many nodes are of type `SERVER` in the network?": [
        "What’s the number of `SERVER` nodes in the network?",
        "How many nodes classified as `SERVER` are there?",
        "Can you provide the count of `SERVER` nodes in this network?",
        "What is the total number of `SERVER` nodes?",
        "How many nodes of the `SERVER` type are present?"
    ],
    "How many nodes are connected to Node_0?": [
        "What is the number of nodes connected to Node_0?",
        "Can you count the nodes linked to Node_0?",
        "How many nodes have connections with Node_0?",
        "How many connections does Node_0 have with other nodes?",
        "What’s the count of nodes attached to Node_0?"
    ],
    "How many links are connected to Node_1?": [
        "What’s the number of links attached to Node_1?",
        "How many links involve Node_1?",
        "Can you provide the count of links connected to Node_1?",
        "How many links is Node_1 connected to?",
        "What is the total number of links associated with Node_1?"
    ],
    "What are the source and destination nodes of Link 2?": [
        "Which nodes are the source and destination for Link 2?",
        "Can you specify the source and destination nodes for Link 2?",
        "What are the nodes connected by Link 2?",
        "Which nodes does Link 2 connect as source and destination?",
        "Identify the source and destination nodes for Link 2."
    ],
    "How many nodes are of type `COMPUTER` in the network?": [
        "What is the total number of `COMPUTER` nodes in the network?",
        "How many `COMPUTER` nodes are there?",
        "Can you count the `COMPUTER` nodes in this network?",
        "What’s the number of `COMPUTER` nodes in the network?",
        "How many nodes are classified as `COMPUTER`?"
    ],
    "How many switches are present in the network?": [
        "What is the number of switches in the network?",
        "How many switches does this network have?",
        "Can you tell me how many switches are in the network?",
        "How many switches are there?",
        "What’s the total count of switches in the network?"
    ]
}