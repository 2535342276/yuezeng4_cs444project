import torch
class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    # def __init__(
    #     self, state=None, hidden=None, target_class_embedding=None, action_probs=None, objbb = None
    # ):
    #     self.state = state
    #     self.hidden = hidden
    #     self.target_class_embedding = target_class_embedding
    #     self.action_probs = action_probs
    #     self.objbb = objbb
    #     self.all_object = None
    def __init__(
        self, state=None, hidden=None, target_class_embedding=None, action_probs=None, objbb=None
    ):
        self.state = state
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs
        self.objbb = objbb
        self.all_object = None

        # Automatically initialize hidden state if it is None
        if hidden is None and state is not None:
            # Assuming state has a batch dimension and using a typical LSTM hidden size
            batch_size = state.size(0)
            hidden_size = 512  # Replace 512 with the actual hidden size used in your LSTM
            zero_hidden = torch.zeros(batch_size, hidden_size).to(state.device)
            zero_cell = torch.zeros(batch_size, hidden_size).to(state.device)
            self.hidden = (zero_hidden, zero_cell)
        else:
            self.hidden = hidden

class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None):

        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
