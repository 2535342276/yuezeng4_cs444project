import models
import agents
import episodes
import optimizers
import argparse

def model_class(class_name):
    if class_name not in models.__all__:
       raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
           class_name, models.__all__))
    return getattr(models, class_name)

def agent_class(class_name):
    if class_name not in agents.__all__:
       raise argparse.ArgumentTypeError("Invalid agent {}; choices: {}".format(
           class_name, agents.__all__))
    return getattr(agents, class_name)
def episode_class(class_name):
    if class_name not in episodes.__all__:
       raise argparse.ArgumentTypeError("Invalid episodes {}; choices: {}".format(
           class_name, episodes.__all__))
    return getattr(episodes, class_name)
def optimizer_class(class_name):
    if class_name not in optimizers.__all__:
       raise argparse.ArgumentTypeError("Invalid optimizer {}; choices: {}".format(
           class_name, optimizers.__all__))
    return getattr(optimizers, class_name)


# import models
# import agents
# import episodes
# import optimizers
# import argparse

# # Your existing functions remain untouched
# def model_class(class_name):
#     print("Available models:", class_name)
#     if class_name not in models.__all__:
#         raise argparse.ArgumentTypeError("Invalid model {}; choices: {}".format(
#             class_name, models.__all__))
#     return getattr(models, class_name)

# def agent_class(class_name):
#     if class_name not in agents.__all__:
#         raise argparse.ArgumentTypeError("Invalid agent {}; choices: {}".format(
#             class_name, agents.__all__))
#     return getattr(agents, class_name)

# def episode_class(class_name):
#     if class_name not in episodes.__all__:
#         raise argparse.ArgumentTypeError("Invalid episodes {}; choices: {}".format(
#             class_name, episodes.__all__))
#     return getattr(episodes, class_name)

# def optimizer_class(class_name):
#     if class_name not in optimizers.__all__:
#         raise argparse.ArgumentTypeError("Invalid optimizer {}; choices: {}".format(
#             class_name, optimizers.__all__))
#     return getattr(optimizers, class_name)

# def print_available_classes():
#     print("Available classes in 'models':")
#     for class_name in models.__all__:
#         try:
#             print(f"  {class_name}: {model_class(class_name)}")
#         except argparse.ArgumentTypeError as e:
#             print(e)

#     print("\nAvailable classes in 'agents':")
#     for class_name in agents.__all__:
#         try:
#             print(f"  {class_name}: {agent_class(class_name)}")
#         except argparse.ArgumentTypeError as e:
#             print(e)

#     print("\nAvailable classes in 'episodes':")
#     for class_name in episodes.__all__:
#         try:
#             print(f"  {class_name}: {episode_class(class_name)}")
#         except argparse.ArgumentTypeError as e:
#             print(e)

#     print("\nAvailable classes in 'optimizers':")
#     for class_name in optimizers.__all__:
#         try:
#             print(f"  {class_name}: {optimizer_class(class_name)}")
#         except argparse.ArgumentTypeError as e:
#             print(e)

# if __name__ == "__main__":
#     print_available_classes()
