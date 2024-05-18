# from __future__ import division

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.net_util import norm_col_init, weights_init

# from .model_io import ModelOutput
# from .model_io import ModelInput
# import copy
        

    
# #####################################################################


# # class BaseModel(nn.Module):
# #     def __init__(self, args):
# #         super(BaseModel, self).__init__()
# #         self.conv1 = nn.Conv2d(args.hidden_state_sz, 64, 1)  # Ensure this matches the channel size of input
# #         self.embed_glove = nn.Linear(args.glove_dim, 64)
# #         self.embed_action = nn.Linear(args.action_space, 10)
# #         self.pointwise = nn.Conv2d(138, 64, 1, 1)

# #         self.lstm = nn.LSTMCell(7 * 7 * 64, args.hidden_state_sz)
# #         self.actor_linear = nn.Linear(args.hidden_state_sz, args.action_space)
# #         self.critic_linear = nn.Linear(args.hidden_state_sz, 1)
# #         self.dropout = nn.Dropout(p=args.dropout_rate)
# #         self.apply(weights_init)

# #     def embedding(self, state, target, action_probs, params):
# #         state = state[None,:,:,:]
# #         action_embedding_input = action_probs

# #         if params is None:
# #             glove_embedding = F.relu(self.embed_glove(target))
# #             glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

# #             action_embedding = F.relu(self.embed_action(action_embedding_input))
# #             action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

# #             image_embedding = F.relu(self.conv1(state))
# #             x = self.dropout(image_embedding)
# #             x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
# #             x = F.relu(self.pointwise(x))
# #             x = self.dropout(x)
# #             out = x.view(x.size(0), -1)

# #         else:
# #             glove_embedding = F.relu(
# #                 F.linear(
# #                     target,
# #                     weight=params["embed_glove.weight"],
# #                     bias=params["embed_glove.bias"],
# #                 )
# #             )

# #             glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

# #             action_embedding = F.relu(
# #                 F.linear(
# #                     action_embedding_input,
# #                     weight=params["embed_action.weight"],
# #                     bias=params["embed_action.bias"],
# #                 )
# #             )
# #             action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

# #             image_embedding = F.relu(
# #                 F.conv2d(
# #                     state, weight=params["conv1.weight"], bias=params["conv1.bias"]
# #                 )
# #             )
# #             x = self.dropout(image_embedding)
# #             x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)

# #             x = F.relu(
# #                 F.conv2d(
# #                     x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
# #                 )
# #             )
# #             x = self.dropout(x)
# #             out = x.view(x.size(0), -1)

# #         return out, image_embedding

# #     def a3clstm(self, embedding, prev_hidden, params):
# #         if params is None:
# #             hx, cx = self.lstm(embedding, prev_hidden)
# #             x = hx
# #             actor_out = self.actor_linear(x)
# #             critic_out = self.critic_linear(x)

# #         else:
# #             hx, cx = self._backend.LSTMCell(
# #                 embedding,
# #                 prev_hidden,
# #                 params["lstm.weight_ih"],
# #                 params["lstm.weight_hh"],
# #                 params["lstm.bias_ih"],
# #                 params["lstm.bias_hh"],
# #             )

# #             # Change for pytorch 1.01
# #             # hx, cx = nn._VF.lstm_cell(
# #             #     embedding,
# #             #     prev_hidden,
# #             #     params["lstm.weight_ih"],
# #             #     params["lstm.weight_hh"],
# #             #     params["lstm.bias_ih"],
# #             #     params["lstm.bias_hh"],
# #             # )

# #             x = hx

# #             critic_out = F.linear(
# #                 x,
# #                 weight=params["critic_linear.weight"],
# #                 bias=params["critic_linear.bias"],
# #             )
# #             actor_out = F.linear(
# #                 x,
# #                 weight=params["actor_linear.weight"],
# #                 bias=params["actor_linear.bias"],
# #             )

# #         return actor_out, critic_out, (hx, cx)


# #     def forward(self, model_input, model_options):

# #         state = model_input.state
# #         (hx, cx) = model_input.hidden

# #         target = model_input.target_class_embedding
# #         action_probs = model_input.action_probs
# #         params = None

# #         x, image_embedding = self.embedding(state, target, action_probs, params)
# #         actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

# #         return ModelOutput(
# #             value=critic_out,
# #             logit=actor_out,
# #             hidden=(hx, cx),
# #             embedding=image_embedding,
# #         )

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def weights_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias)

# class BaseModel(nn.Module):
#     def __init__(self, args):
#         action_space = args.action_space
#         target_embedding_sz = args.glove_dim
#         resnet_embedding_sz = args.hidden_state_sz
#         hidden_state_sz = args.hidden_state_sz
#         super(BaseModel, self).__init__()
#         self.conv1 = nn.Conv2d(args.hidden_state_sz, 64, 1)  # Ensure this matches the channel size of input
#         self.embed_glove = nn.Linear(args.glove_dim, 64)
#         self.embed_action = nn.Linear(args.action_space, 10)
#         self.pointwise = nn.Conv2d(138, 64, 1, 1)
        
#         lstm_input_sz = 7 * 7 * 64
#         self.hidden_state_sz = hidden_state_sz
#         self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
#         num_outputs = action_space
#         self.critic_linear = nn.Linear(hidden_state_sz, 1)
#         self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

#         self.lstm = nn.LSTMCell(7 * 7 * 64, args.hidden_state_sz)
#         self.actor_linear = nn.Linear(args.hidden_state_sz, args.action_space)
#         self.critic_linear = nn.Linear(args.hidden_state_sz, 1)
#         self.fc_final = nn.Linear(args.hidden_state_sz, 128)  # Adjusted to 128 for compatibility with MoCo
#         self.dropout = nn.Dropout(p=args.dropout_rate)

#         self.output_features_size = 128  # This is the size of the output from fc_final

#         self.apply(weights_init)
# #￥￥￥￥￥￥￥￥￥￥￥
#     def embedding(self, state, target, action_probs, params):
#         state = state[None,:,:,:]
#         action_embedding_input = action_probs

#         if params is None:
#             glove_embedding = F.relu(self.embed_glove(target))
#             glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

#             action_embedding = F.relu(self.embed_action(action_embedding_input))
#             action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

#             image_embedding = F.relu(self.conv1(state))
#             x = self.dropout(image_embedding)
#             x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
#             x = F.relu(self.pointwise(x))
#             x = self.dropout(x)
#             out = x.view(x.size(0), -1)

#         else:
#             glove_embedding = F.relu(
#                 F.linear(
#                     target,
#                     weight=params["embed_glove.weight"],
#                     bias=params["embed_glove.bias"],
#                 )
#             )

#             glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

#             action_embedding = F.relu(
#                 F.linear(
#                     action_embedding_input,
#                     weight=params["embed_action.weight"],
#                     bias=params["embed_action.bias"],
#                 )
#             )
#             action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

#             image_embedding = F.relu(
#                 F.conv2d(
#                     state, weight=params["conv1.weight"], bias=params["conv1.bias"]
#                 )
#             )
#             x = self.dropout(image_embedding)
#             x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)

#             x = F.relu(
#                 F.conv2d(
#                     x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
#                 )
#             )
#             x = self.dropout(x)
#             out = x.view(x.size(0), -1)

#         return out, image_embedding
#     def a3clstm(self, embedding, prev_hidden, params):
#         if params is None:
#             hx, cx = self.lstm(embedding, prev_hidden)
#             x = hx
#             actor_out = self.actor_linear(x)
#             critic_out = self.critic_linear(x)

#         else:
#             hx, cx = self._backend.LSTMCell(
#                 embedding,
#                 prev_hidden,
#                 params["lstm.weight_ih"],
#                 params["lstm.weight_hh"],
#                 params["lstm.bias_ih"],
#                 params["lstm.bias_hh"],
#             )

#             # Change for pytorch 1.01
#             # hx, cx = nn._VF.lstm_cell(
#             #     embedding,
#             #     prev_hidden,
#             #     params["lstm.weight_ih"],
#             #     params["lstm.weight_hh"],
#             #     params["lstm.bias_ih"],
#             #     params["lstm.bias_hh"],
#             # )

#             x = hx

#             critic_out = F.linear(
#                 x,
#                 weight=params["critic_linear.weight"],
#                 bias=params["critic_linear.bias"],
#             )
#             actor_out = F.linear(
#                 x,
#                 weight=params["actor_linear.weight"],
#                 bias=params["actor_linear.bias"],
#             )

#         return actor_out, critic_out, (hx, cx)



#     def forward(self, model_input):
#         if model_input.target_class_embedding is None:
#             print("Warning: target_class_embedding is None, this may lead to errors in processing.")
#         # Extract attributes from model_input using the correct names
#         state = model_input.state
#         target = model_input.target_class_embedding
#         action_probs = model_input.action_probs
#         hidden = model_input.hidden

#         # Embed target and action probabilities
#         glove_embedding = F.relu(self.embed_glove(target))
#         action_embedding = F.relu(self.embed_action(action_probs))

#         # Reshape embeddings to match convolutional feature maps
#         glove_reshaped = glove_embedding.view(-1, 64, 1, 1).expand(-1, -1, 7, 7)
#         action_reshaped = action_embedding.view(-1, 10, 1, 1).expand(-1, -1, 7, 7)

#         # Initial state processing with convolution
#         state_features = F.relu(self.conv1(state))
#         combined_features = torch.cat([state_features, glove_reshaped, action_reshaped], dim=1)
#         combined_features = self.pointwise(combined_features)

#         # Flatten features for LSTM
#         lstm_input = combined_features.view(-1, 7 * 7 * 64)
#         hx, cx = self.lstm(lstm_input, hidden)

#         # Final linear transformation
#         output_features = self.fc_final(hx)

#         return output_features, (hx, cx)


#     def embedding(self, inputs):
#         state, target, action_probs = inputs
#         state = state.unsqueeze(0)  # Ensure input has batch dimension if needed
#         x = F.relu(self.conv1(state))
#         glove_embedding = F.relu(self.embed_glove(target)).view(1, 64, 1, 1).expand(-1, -1, 7, 7)
#         action_embedding = F.relu(self.embed_action(action_probs)).view(1, 10, 1, 1).expand(-1, -1, 7, 7)
#         x = torch.cat([x, glove_embedding, action_embedding], dim=1)
#         x = F.relu(self.pointwise(x))
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)  # Linear layer to get embeddings of size 128
#         return F.normalize(x, dim=1)  # Normalize for contrastive loss

# ################################################################################



# # class MoCo(nn.Module):
# #     def __init__(self, base_model, dim=128, K=65536, m=0.999, T=0.07):
# #         super(MoCo, self).__init__()
# #         self.base_model = base_model
# #         if not isinstance(base_model, nn.Module):
# #             raise TypeError("base_model should be an instance of torch.nn.Module")

# #         self.K = K
# #         self.m = m
# #         self.T = T
# #         self.dim = dim

# #         self.encoder_q = base_model(num_classes=dim)
# #         self.encoder_k = copy.deepcopy(self.encoder_q)
# #         for param in self.encoder_k.parameters():
# #             param.requires_grad = False  # Freeze the key encoder

# #         self.register_buffer("queue", torch.randn(dim, K))
# #         self.queue = nn.functional.normalize(self.queue, dim=0)
# #         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

# #     def forward(self, input_q, input_k):
# #         if not isinstance(input_q, torch.Tensor) or not isinstance(input_k, torch.Tensor):
# #             raise TypeError("Both input_q and input_k must be torch.Tensor instances")

# #         q = self.encoder_q(input_q)
# #         q = nn.functional.normalize(q, dim=1)

# #         with torch.no_grad():
# #             k = self.encoder_k(input_k)
# #             k = nn.functional.normalize(k, dim=1)

# #         # Compute logits
# #         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
# #         l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
# #         logits = torch.cat([l_pos, l_neg], dim=1) / self.T
# #         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input_q.device)

# #         # Update queue
# #         self._dequeue_and_enqueue(k)

# #         return logits, labels

# #     @torch.no_grad()
# #     def _dequeue_and_enqueue(self, keys):
# #         batch_size = keys.shape[0]
# #         ptr = int(self.queue_ptr)
# #         self.queue[:, ptr:ptr + batch_size] = keys.T
# #         ptr = (ptr + batch_size) % self.K
# #         self.queue_ptr[0] = ptr

# #     @torch.no_grad()
# #     def _momentum_update_key_encoder(self):
# #         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
# #             param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)


# class MoCo(nn.Module):
#     def __init__(self, base_model, dim=128, K=65536, m=0.999, T=0.07):
#         super(MoCo, self).__init__()
#         self.K = K
#         self.m = m
#         self.T = T
#         self.dim = dim

#         self.encoder_q = base_model
#         self.encoder_k = copy.deepcopy(self.encoder_q)

#         # Freeze the parameters of the key encoder
#         for param in self.encoder_k.parameters():
#             param.requires_grad = False

#         # Define the fully connected layers for query and key encoders
#         self.fc_q = nn.Linear(base_model.output_features_size, dim)  # Adjust 'output_features_size' as needed
#         self.fc_k = nn.Linear(base_model.output_features_size, dim)  # Adjust 'output_features_size' as needed

#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

#     def forward(self, input_q, input_k):
#         q = self.fc_q(self.encoder_q(input_q))
#         q = nn.functional.normalize(q, dim=1)

#         with torch.no_grad():
#             self._momentum_update_key_encoder()  # Ensure this method is implemented
#             k = self.fc_k(self.encoder_k(input_k))
#             k = nn.functional.normalize(k, dim=1)

#         # Compute logits
#         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#         l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
#         logits = torch.cat([l_pos, l_neg], dim=1) / self.T
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input_q.device)

#         # Update queue
#         self._dequeue_and_enqueue(k)

#         return logits, labels

#     # Make sure to implement or define the methods _momentum_update_key_encoder and _dequeue_and_enqueue as used in forward method.
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, keys):
#         batch_size = keys.shape[0]
#         ptr = int(self.queue_ptr)
#         self.queue[:, ptr:ptr + batch_size] = keys.T
#         ptr = (ptr + batch_size) % self.K
#         self.queue_ptr[0] = ptr

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)





from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput
from .model_io import ModelInput
import copy
        


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_embedding_sz, 64)
        self.embed_action = nn.Linear(action_space, 10)

        pointwise_in_channels = 138

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_probs, params):
        state = state[None,:,:,:]
        action_embedding_input = action_probs

        if params is None:
            glove_embedding = F.relu(self.embed_glove(target))
            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(self.conv1(state))
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = F.relu(self.pointwise(x))
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:
            glove_embedding = F.relu(
                F.linear(
                    target,
                    weight=params["embed_glove.weight"],
                    bias=params["embed_glove.bias"],
                )
            )

            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(
                F.linear(
                    action_embedding_input,
                    weight=params["embed_action.weight"],
                    bias=params["embed_action.bias"],
                )
            )
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(
                F.conv2d(
                    state, weight=params["conv1.weight"], bias=params["conv1.bias"]
                )
            )
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)

            x = F.relu(
                F.conv2d(
                    x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
                )
            )
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden, params):
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:
            hx, cx = self._backend.LSTMCell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        params = None

        x, image_embedding = self.embedding(state, target, action_probs, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
