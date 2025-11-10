import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from make_graph import extract_graph_features, EmbedValue
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import matplotlib.pyplot as plt
# data_root = "./raw_data/"
# csv_files = ["data0000.csv"]
# output_root = "./output/"
# data_list = []
# data_list2 = []

# "meta-llama/Llama-2-7b-hf": {"source": "huggingface"},
# "bigscience/bloom-560m": {"source": "huggingface"},
# "google/gemma-2b": {"source": "huggingface"},
# "google/gemma-2-2b": {"source": "huggingface"},
# "microsoft/Phi-3-mini-128k-instruct": {"source": "huggingface"},
# "Qwen/Qwen2-0.5B": {"source": "huggingface"},

# GPUs = {
#     "GPU 0: NVIDIA L4": "L4",
#     "GPU 0: Tesla T4": "T4",
#     "GPU 0: NVIDIA A100-SXM4-40GB": "A100",
# }
# above is for cluster-inference tests

bits = {"torch.float16": 16, "torch.float32": 32}

def unify(data_list):
    num_features = 9  # Fixed number of features per node
    max_values = torch.zeros(num_features)
    for data in data_list:
        graph_max = torch.max(data.x, dim=0).values
        max_values = torch.max(max_values, graph_max)
    max_values = max_values + 1e-10  #avoid dividing zero
    for data in data_list:
        data.x = data.x / max_values  
    return data_list

def unify_feature(global_feature_list):
    #To be implemented
    pass



def process_csv_files(
    data_root="./raw_data/", csv_files=["data0000.csv"], output_root="./output/",show=False
):
    data_list = []
    data_list2 = []
    global_list = []

    example_to_show={'nodes':None,'edges':None}
    example_index=2
    current_index=0

    for csv in csv_files:
        with open(data_root + csv,encoding='utf-8') as f:
            for line in f.readlines():
                line = line.rstrip()
                items = line.split(",")
                gpu='rockchip_rk3588'
                '''
                gpu = str(items[0])
                print(gpu=="RK3588-NPU")
                if gpu == "A100":
                    gpu = "nvidia_A100_40G"
                elif gpu == "T4":
                    gpu = "nvidia_T4"
                    continue
                elif gpu == "L4":
                    gpu = "nvidia_L4"
                elif gpu == "RK3588-NPU":
                    gpu = "rockchip_rk3588"
                '''
                gpu_num = float(items[1])
   
                llm = str(items[2])
                if llm == "internlm":
                     llm = "internlm/internlm2-chat-1_8b"
                elif llm == "qwen2-1.5b":
                    llm = "Qwen/Qwen2-1.5B"
                elif llm == "tinyllama":
                     llm = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                elif llm == "qwen1.5-0.5b":
                     llm = "Qwen/Qwen1.5-0.5B"
                # elif llm == "phi3":
                #     llm = "microsoft/Phi-3-mini-128k-instruct"
                # elif llm == "qwen2":
                #     llm = "Qwen/Qwen2-0.5B"
  

                bit = float(items[3])
                act = str(items[4])
                hidden = float(items[5])
                inter = float(items[6])
                layer = float(items[7])
                head = float(items[8])
                vob = float(items[9])
                batch = float(items[10])
                prompt_length = float(items[11])
                token_length = float(items[12])
                latency = float(items[13])
                if items[14]=='0' or items[14]=='0.0':
                    print('abandon this line because of recording failure')
                else:
                    energy = float(items[14])

                width = None
                if bit == 32:
                    width = "FP32"
                elif bit == 16:
                    width = "FP16"
                elif bit == 8:
                    width = "INT8"
                else:
                    width = "INT4"

                stage_temp = None
                if token_length > 1:
                    stage_temp = "decode"
                else:
                    stage_temp = "prefill"

                inference_config = {
                    "stage": stage_temp,
                    "batch_size": batch,
                    "seq_length": prompt_length,
                    "gen_length": token_length,
                    "w_quant": width,
                    "a_quant": width,
                    "kv_quant": width,
                    "use_flashattention": False,
                    "activation": act,
                    "hidden_size": hidden,
                    "inter_size": inter,
                    "layer_num": layer,
                    "head_num": head,
                    "vob_size": vob,
                }

                #nodes, edges, global_f = extract_graph_features(
                #    llm, gpu, inference_config
                #)
                nodes, edges, global_f = extract_graph_features(
                    llm, gpu, inference_config
                )
                edge_index = torch.from_numpy(np.array(np.where(edges > 0))).type(
                    torch.long
                )
                node_features = np.array(nodes, dtype=np.float32)

                # if show:
                #     print('///////////////')
                #     print(items)
                #     print(node_features)
                #     print(global_f)
                

                x = torch.from_numpy(node_features).type(torch.float)
                #print('x:',x[1:2])
                y = torch.FloatTensor([latency])

                #print('y:',y[1:2])

                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
                y2 = torch.FloatTensor([energy]) #??????
                #print('y2:',y2[1:2])
                data2 = Data(x=x, edge_index=edge_index, y=y2)
                data_list2.append(data2)
                global_feature = torch.from_numpy(global_f).type(torch.float)
                #print('gloabal_f:',global_feature[1:2])

                #print(f"x: {x.size()}")
                #print(f"y: {y.size()}")
                #print(f"e: {edge_index.size()}")
                #print(f"g: {global_feature.size()}")
                # if current_index==example_index:
                #     example_to_show['nodes']=node_features
                #     example_to_show['edges']=edge_index
                #     example_to_show['ys']=y
                #     example_to_show['global_feature']=global_feature

                global_list.append(global_feature)
                current_index+=1
    # if show:
    #     print(example_to_show['nodes'])
    #     print(example_to_show['edges'])
    #     print(example_to_show['ys'])
    #     print(example_to_show['global_feature'])
        #draw_graph_with_arrows(example_to_show['nodes'], example_to_show['edges'], labels=None)       

    if (
        len(data_list) != len(data_list2)
        or len(data_list) != len(global_list)
        or len(global_list) != len(data_list2)
    ):
        print("length error")

    


    data_list=unify(data_list)
    data_list2=unify(data_list2)
    #global_list=unify_feature(global_list)


    torch.save(data_list, output_root + "latency_data.pt")
    torch.save(data_list2, output_root + "energy_data.pt")
    torch.save(global_list, output_root + "global_feature.pt")
    #print(global_list[1:3])



    with open(output_root + "length.txt", mode="w", encoding="utf-8") as ref:
        ref.write(str(len(data_list)))
        ref.close()



def draw_graph_with_arrows(node_features, edge_index, labels=None):
    """
    Draws a directed graph with arrows connecting nodes.

    Parameters:
        node_features (torch.Tensor): Node features tensor (N x F).
        edge_index (torch.Tensor): Edge connections (2 x E).
        labels (list or torch.Tensor, optional): Labels for nodes (N).
    """
    # Create a directed graph using NetworkX
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for i, features in enumerate(node_features):
        label = f"{i}: {tuple(features.tolist())}" if labels is None else f"{i} ({labels[i].item()})"
        G.add_node(i, label=label)
    
    # Add edges
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    # Position nodes for visualization
    pos = nx.spring_layout(G, seed=42)  # Spring layout positions
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=20)
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=10)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)
    plt.title("Directed Graph Visualization with Arrows")
    plt.show()


def main():
    out_dir = "./"
    process_csv_files(data_root="./raw_data/xxx", csv_files=["xxx.csv"], output_root="./processed_data/xxx_code_trace_train_",show=False)


if __name__ == "__main__":

    main()
