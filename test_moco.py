import torch
import torch.nn.functional as F
from models.basemodel import BaseModel, MoCo, ModelInput
class Args:
    action_space = 10      
    glove_dim = 100         
    hidden_state_sz = 512    
    dropout_rate = 0.1      
    feature_dim = 128
    queue_size = 1024
    momentum = 0.99
    temperature = 0.07
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()
moco = MoCo(BaseModel, args, dim=args.feature_dim, K=args.queue_size, m=args.momentum, T=args.temperature).to(args.device)
moco.train()

batch_size = 10
channels, height, width = 3, 224, 224
images = torch.randn((batch_size, channels, height, width)).to(args.device)



hiddenfake = (torch.zeros(1, 512).to(args.device), torch.zeros(1, 512).to(args.device))  # LSTM hidden state size 512
dummy_target_embedding = torch.zeros(1, 100).to(args.device)  # target class embedding size 100
dummy_action_probs = torch.zeros(1, 10).to(args.device)  # action space size 10

model_input_q = ModelInput(images, hiddenfake, dummy_target_embedding, dummy_action_probs)
model_input_k = ModelInput(images, hiddenfake, dummy_target_embedding, dummy_action_probs)

# forward propagation
logits, labels = moco(model_input_q, model_input_k)

loss = F.cross_entropy(logits, labels)
print("Contrastive loss:", loss.item())
print("Queue pointer before update:", moco.queue_ptr)
moco._dequeue_and_enqueue(torch.randn((batch_size, args.feature_dim)).to(args.device))
print("Queue pointer after update:", moco.queue_ptr)

assert not torch.isnan(loss), "loss is NaN"
assert moco.queue_ptr.item() != 0, "queue pointer did not update"

print("MoCo test passed.")
