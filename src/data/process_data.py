def format_data(
        agents, utterances, utterance_labels, UTT, DH, ST, DA_E, max_tokens=512) -> list:
    data = []
    for i, (agent, utterance, _) in enumerate(zip(agents, utterances, utterance_labels)):
        cur_data = ""
        if DH and i > 0:
            cur_data += data[-1] + ' '
            if DA_E:
                cur_data += f'<<{",".join(utterance_labels[i - 1])}>> '
            cur_data += '<<TURN>> '
        if ST:
            cur_data += f'<<{agent}>> '
        cur_data += utterance
        # The following slice decreased runtime from 685s to 6s
        if DH:
            data.append(" ".join(cur_data.split()[-max_tokens:]))
        else:
            data.append(cur_data)

    # Remove utterances if trying to predict future dialogue acts
    if not UTT:
        for i in range(len(data)):
            data[i] = data[i][:data[i].rfind("<<TURN>>") + 8]

    return data


