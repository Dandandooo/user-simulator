import random

class SpandanaHistories:
    def __init__(self, data: list[list[dict]] or None, utt: bool, st: bool, dh: bool, dae: bool, predict: bool = False):
        self.data = data
        self.prompter = SpandanaPrompt(utt, st, dh, dae, predict)

    def __call__(self, data_override = None):
        data = data_override if data_override is not None else self.data
        if data:
            raise ValueError("You must pass a dataset")
        return self.prompter(random.choice(data))

    def iter_n(self, n: int, dataset = None):
        for _ in range(n):
            yield self(dataset)

class SpandanaPrompt:
    def __init__(self, utt: bool, st: bool, dh: bool, dae: bool, predict: bool = False):
        self.Utt = utt
        self.ST = st
        self.DH = dh
        self.DA_E = dae

        self.predict = predict

    def __call__(self, task: list[dict], length=None) -> tuple[str, list]:
        task = [d for d in task if d["turn_action"] == "dialogue"]
        if length is None:
            length = random.randint(1, len(task)-1-self.predict)

        # Returns utterance/history and dialogue acts of current turn
        return self._example(task, length)

    def _example(self, task, to_turn) -> tuple[str, list]:
        return self._history(task, to_turn), self._turn_info(task[to_turn])[2]

    def _history(self, task, to_turn):
        if self.DH:
            return " <<TURN>> ".join([*map(self._turn_to_line, task[:to_turn]),
                                      self._turn_to_line(task[to_turn], last_turn=True)])
        return self._turn_to_line(task[to_turn])

    @staticmethod
    def _turn_info(turn: dict) -> tuple:
        user = "Commander" if turn["COMMANDER"]["action"] == "dialogue" else "Follower"
        utterance = turn[user.upper()]["utterance"]
        das = turn[user.upper()]["das"]
        return user, utterance, das

    def _turn_to_line(self, turn: dict, last_turn: bool = False) -> str or list[str]:
        user, utterance, das = self._turn_info(turn)
        das = ",".join(das)

        prompt = ""

        if self.ST:
            prompt += f"<<{user}>> "

        if self.Utt:
            prompt += utterance

        if self.DA_E ^ last_turn:
            prompt += f" <<{das}>>"

        return prompt
