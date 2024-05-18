# from __future__ import print_function, division
# from models.model_io import ModelInput
# from models.model_io import ModelOptions
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
# import random
# import ctypes
# import setproctitle
# import time

# import numpy as np
# import torch
# import torch.multiprocessing as mp
# from tensorboardX import SummaryWriter

# from utils import flag_parser

# from utils.class_finder import model_class, agent_class, optimizer_class
# from utils.net_util import ScalarMeanTracker
# from main_eval import main_eval,main_eval_unseen,main_eval_seen

# from runners import nonadaptivea3c_train, nonadaptivea3c_val, savn_train, savn_val,savn_train_seen
# from runners import nonadaptivea3c_train_seen,nonadaptivea3c_val_unseen


# os.environ["OMP_NUM_THREADS"] = "1"

# def main():
#     setproctitle.setproctitle("Train/Test Manager")
#     args = flag_parser.parse_arguments()
#     print(args)
#     # Determine the target function based on the model type and evaluation settings
#     if args.model == "SAVN":
#         if args.zsd:
#             print("use zsd setting !")
#             args.learned_loss = True
#             args.num_steps = 6
#             target = savn_val if args.eval else savn_train_seen
#         else:
#             args.learned_loss = True
#             args.num_steps = 6
#             target = savn_val if args.eval else savn_train
#     else:
#         if args.zsd:
#             print("use zsd setting !")
#             args.learned_loss = False
#             args.num_steps = 50
#             target = nonadaptivea3c_val_unseen if args.eval else nonadaptivea3c_train_seen
#         else:
#             args.learned_loss = False
#             args.num_steps = 50
#             target = nonadaptivea3c_val if args.eval else nonadaptivea3c_train
#     # added
#     # create_shared_model = model_class(args.model)
#     # MoCo
#     create_shared_model = model_class(args.model, args)
#     init_agent = agent_class(args.agent_type)
#     optimizer_type = optimizer_class(args.optimizer)

#     if args.eval:
#         if args.zsd:
#             print("Evaluate Unseen Classes !")
#             main_eval_unseen(args, create_shared_model, init_agent)
#             print("#######################################################")
#             print("Evaluate Seen Classes !")
#             main_eval_seen(args, create_shared_model, init_agent)
#             return
#         else:
#             main_eval(args, create_shared_model, init_agent)
#             return

#     start_time = time.time()
#     local_start_time_str = time.strftime(
#         "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
#     )
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)

#     if args.log_dir is not None:
#         tb_log_dir = args.log_dir + "/" + args.title + "-" + local_start_time_str
#         log_writer = SummaryWriter(log_dir=tb_log_dir)
#     else:
#         log_writer = SummaryWriter(comment=args.title)

#     if args.gpu_ids == -1:
#         args.gpu_ids = [-1]
#     else:
#         torch.cuda.manual_seed(args.seed)
#         mp.set_start_method("spawn")
    
#     args = flag_parser.parse_arguments()
#     data_q = torch.randn(1, 3, 224, 224, device='cuda')
#     data_k = torch.randn(1, 3, 224, 224, device='cuda')
#     model_options = ModelOptions() 
#     # Assuming ModelInput takes an image tensor and possibly other parameters
#     dummy_input_q = ModelInput(state=data_q)
#     dummy_input_k = ModelInput(state=data_k)
#     logits, labels = create_shared_model(dummy_input_q, dummy_input_k)
#     # Now pass both the inputs and the model_options
#  # Update this line based on actual signature

#     # shared_model = create_shared_model(dummy_input_q, dummy_input_k, model_options)

#     train_total_ep = 0
#     n_frames = 0

#     if shared_model is not None:
#         shared_model.share_memory()
#         optimizer = optimizer_type(
#             filter(lambda p: p.requires_grad, shared_model.parameters()), args
#         )
#         optimizer.share_memory()
#         print(shared_model)
#     else:
#         assert (
#             args.agent_type == "RandomNavigationAgent"
#         ), "The model is None but agent is not random agent"
#         optimizer = None

#     processes = []

#     end_flag = mp.Value(ctypes.c_bool, False)

#     train_res_queue = mp.Queue()

#     for rank in range(0, args.workers):
#         p = mp.Process(
#             target=target,
#             args=(
#                 rank,
#                 args,
#                 create_shared_model,
#                 shared_model,
#                 init_agent,
#                 optimizer,
#                 train_res_queue,
#                 end_flag,
#             ),
#         )
#         p.start()
#         processes.append(p)
#         time.sleep(0.1)

#     print("Train agents created.")

#     train_thin = args.train_thin
#     train_scalars = ScalarMeanTracker()

#     save_entire_model = 0
#     try:
#         time_start = time.time()
#         reward_avg_list = []
#         ep_length_avg_list=[]
#         while train_total_ep < args.max_ep:
#             train_result = train_res_queue.get()
#             reward_avg_list.append(train_result["total_reward"])
#             train_scalars.add_scalars(train_result)
#             train_total_ep += 1
#             n_frames += train_result["ep_length"]
#             ep_length_avg_list.append(train_result["ep_length"])
#             if (train_total_ep % train_thin) == 0:
#                 log_writer.add_scalar("n_frames", n_frames, train_total_ep)
#                 tracked_means = train_scalars.pop_and_reset()
#                 for k in tracked_means:
#                     log_writer.add_scalar(
#                         k + "/train", tracked_means[k], train_total_ep
#                     )

#             if (train_total_ep % args.ep_save_freq) == 0:
#                 print(n_frames)
#                 if not os.path.exists(args.save_model_dir):
#                     os.makedirs(args.save_model_dir)
#                 state_to_save = shared_model.state_dict()
#                 save_path = os.path.join(
#                     args.save_model_dir,
#                     "{0}_{1}_{2}_{3}.dat".format(
#                         args.model, train_total_ep, n_frames, local_start_time_str
#                     ),
#                 )
#                 torch.save(state_to_save, save_path)
#                 if os.path.exists(save_path):
#                     print("file saved successful to：", save_path)
#                 else:
#                     print("fail to save file")
#                 save_entire_model += 1
#                 if (save_entire_model % 5) == 0:
#                     state = {
#                         'epoch': train_total_ep,
#                         'state_dict': shared_model.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                     }

#                     save_model_path = os.path.join(
#                         args.save_model_dir,
#                         "{0}_{1}_{2}.tar".format(
#                             args.model, train_total_ep, local_start_time_str
#                         ),
#                     )
#                     torch.save(state,save_model_path)
#                     save_entire_model=0

#             if train_total_ep % 100 == 0:
#                 time_end = time.time()
#                 seconds = round(time_end - time_start)
#                 m, s = divmod(seconds, 60)
#                 h, m = divmod(m, 60)
#                 reward_avg = sum(reward_avg_list)/len(reward_avg_list)
#                 ep_length_avg = sum(ep_length_avg_list)/len(ep_length_avg_list)
#                 print("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]"
#                       .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
#                 reward_avg_list = []
#                 ep_length_avg_list = []
#                 save_path = os.path.join( args.save_model_dir,"{0}_{1}.txt".format(args.model,local_start_time_str))
#                 print("################ save_model_dir: ",args.save_model_dir)
#                 print("################ save_path: ", save_path)
#                 save_dir = os.path.dirname(save_path)  
#                 if not os.path.exists(save_dir):
#                     os.makedirs(save_dir, exist_ok=True)  

#                 f = open(save_path, "a")
#                 if train_total_ep == 100:
#                     f.write(str(args))
#                     f.write("\n")
#                     f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
#                             .format(train_total_ep, args.max_ep, h, m, s, reward_avg, ep_length_avg))
#                 else:
#                     f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
#                           .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
#                 f.close()
#     finally:
#         log_writer.close()
#         end_flag.value = True
#         for p in processes:
#             time.sleep(0.1)
#             p.join()


# if __name__ == "__main__":
#     main()

from __future__ import print_function, division
from models.model_io import ModelInput, ModelOptions
import os
import re
import random
import ctypes
import setproctitle
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


from utils import flag_parser
from utils.class_finder import model_class, agent_class, optimizer_class
from utils.net_util import ScalarMeanTracker
from main_eval import main_eval, main_eval_unseen, main_eval_seen
from runners import (
    nonadaptivea3c_train, nonadaptivea3c_val, savn_train, savn_val, savn_train_seen,
    nonadaptivea3c_train_seen, nonadaptivea3c_val_unseen
)
from datasets.data import get_data
from datasets.scene_util import get_scenes
from datasets.constants import (
    KITCHEN_OBJECT_CLASS_LIST,
    LIVING_ROOM_OBJECT_CLASS_LIST,
    BEDROOM_OBJECT_CLASS_LIST,
    BATHROOM_OBJECT_CLASS_LIST,
    FULL_OBJECT_CLASS_LIST,
)

os.environ["OMP_NUM_THREADS"] = "1"
def worker(rank, args, create_shared_model, shared_model, init_agent, optimizer, train_res_queue, end_flag, features, model_options):
    while not end_flag.value:
        try:
            model_input = ModelInput(state=features)
            output = shared_model(model_input, model_options)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Worker {rank} has an error: {e}")
            break


    
def main():
    setproctitle.setproctitle("Train/Test Manager")
    args = flag_parser.parse_arguments()
    print(args)

    if args.model == "SAVN":
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val if args.eval else savn_train_seen
        else:
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val if args.eval else savn_train
    else:
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val_unseen if args.eval else nonadaptivea3c_train_seen
        else:
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val if args.eval else nonadaptivea3c_train

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)
    optimizer_type = optimizer_class(args.optimizer)

    if args.eval:
        if args.zsd:
            print("Evaluate Unseen Classes !")
            main_eval_unseen(args, create_shared_model, init_agent)
            print("#######################################################")
            print("Evaluate Seen Classes !")
            main_eval_seen(args, create_shared_model, init_agent)
            return
        else:
            main_eval(args, create_shared_model, init_agent)
            return

    start_time = time.time()
    local_start_time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(start_time))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.log_dir is not None:
        tb_log_dir = args.log_dir + "/" + args.title + "-" + local_start_time_str
        log_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        log_writer = SummaryWriter(comment=args.title)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

        # main.py

    import re


    room_types = ['kitchen', 'living_room', 'bedroom', 'bathroom']
    room_mapping = {'kitchen': 0, 'living_room': 1, 'bedroom': 2, 'bathroom': 3}

    scenes = []
    possible_targets = []
    targets = []
    rooms = []

    def get_data(room, scenes_str):
        scene_list = get_scenes(scenes_str)
        single_possible_targets = FULL_OBJECT_CLASS_LIST  # 根据房间和场景生成
        single_targets = [
        KITCHEN_OBJECT_CLASS_LIST,
        LIVING_ROOM_OBJECT_CLASS_LIST,
        BEDROOM_OBJECT_CLASS_LIST,
        BATHROOM_OBJECT_CLASS_LIST,
    ]  
        single_rooms = [room] * len(scene_list)

        return scene_list, single_possible_targets, single_targets, single_rooms

    for room in room_types:
        if room in room_mapping:
            index = room_mapping[room]
            scene_ids_list = list(range(1, 21))
            scenes_str = "[{}]+{}-{}".format(index, scene_ids_list[0], scene_ids_list[-1])
            # scenes_str = "[{}]+[{}-{}]".format(index, scene_ids_list[0], scene_ids_list[-1])

            print(f"Processing room: {room} with scenes: {scenes_str}")
            single_scenes, single_possible_targets, single_targets, single_rooms = get_data(room, scenes_str)

            scenes.extend(single_scenes)
            possible_targets.extend(single_possible_targets)
            targets.extend(single_targets)
            rooms.extend(single_rooms)

    print("possible_targets: ", possible_targets, "\n")
    print("targets: ",targets,"\n")
    print("rooms: ",rooms,"\n")
    encoder = OneHotEncoder(sparse=False)
    possible_targets_encoded = encoder.fit_transform(np.array(possible_targets).reshape(-1, 1))

    rooms_encoded = encoder.fit_transform(np.array(rooms).reshape(-1, 1))

    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(possible_targets_encoded)

    cluster_labels = kmeans.labels_
    def integrate_clusters_to_training(cluster_labels, other_model_inputs):
        print("Cluster Labels:", cluster_labels)
        enhanced_features = np.hstack([other_model_inputs, cluster_labels.reshape(-1, 1)])
        return enhanced_features

    other_model_inputs = np.random.rand(len(possible_targets_encoded), 10)  
    enhanced_features = integrate_clusters_to_training(cluster_labels, other_model_inputs)




    enhanced_features_tensor = torch.tensor(enhanced_features, dtype=torch.float32)


    split_enhanced_features = torch.chunk(enhanced_features_tensor, args.workers)
    shared_model = create_shared_model(args)

    
    train_total_ep = 0
    n_frames = 0

    if shared_model is not None:
        shared_model.share_memory()
        optimizer = optimizer_type(
            filter(lambda p: p.requires_grad, shared_model.parameters()), args
        )
        optimizer.share_memory()
        print(shared_model)
    else:
        assert (
            args.agent_type == "RandomNavigationAgent"
        ), "The model is None but agent is not random agent"
        optimizer = None

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)

    train_res_queue = mp.Queue()
    
    model_options = ModelOptions() 

    for rank in range(args.workers):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,  
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
                split_enhanced_features[rank],
                model_options 
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)


    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    save_entire_model = 0
    try:
        time_start = time.time()
        reward_avg_list = []
        ep_length_avg_list=[]
        while train_total_ep < args.max_ep:
            train_result = train_res_queue.get()
            reward_avg_list.append(train_result["total_reward"])
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            ep_length_avg_list.append(train_result["ep_length"])
            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/train", tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:
                print(n_frames)
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    "{0}_{1}_{2}_{3}.dat".format(
                        args.model, train_total_ep, n_frames, local_start_time_str
                    ),
                )
                torch.save(state_to_save, save_path)

                save_entire_model += 1
                if (save_entire_model % 5) == 0:
                    state = {
                        'epoch': train_total_ep,
                        'state_dict': shared_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }

                    save_model_path = os.path.join(
                        args.save_model_dir,
                        "{0}_{1}_{2}.tar".format(
                            args.model, train_total_ep, local_start_time_str
                        ),
                    )
                    torch.save(state,save_model_path)
                    save_entire_model=0

            if train_total_ep % 100 == 0:
                time_end = time.time()
                seconds = round(time_end - time_start)
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                reward_avg = sum(reward_avg_list)/len(reward_avg_list)
                ep_length_avg = sum(ep_length_avg_list)/len(ep_length_avg_list)
                print("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]"
                      .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
                reward_avg_list = []
                ep_length_avg_list = []
                save_path = os.path.join( args.save_model_dir,"{0}_{1}.txt".format(args.model,local_start_time_str))
                f = open(save_path, "a")
                if train_total_ep == 100:
                    f.write(str(args))
                    f.write("\n")
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                            .format(train_total_ep, args.max_ep, h, m, s, reward_avg, ep_length_avg))
                else:
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                          .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
                f.close()
    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


if __name__ == "__main__":
    main()