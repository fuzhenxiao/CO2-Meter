import numpy as np
#import re
from hardware import GPU_type
from llm_config.llm_base import kernel_type, avaliable_model_ids_sources, llm_type, act_type
from extract_feature import ModelAnalyzer



class EmbedValue:
    # int value embedding
    @staticmethod
    def embed_int(x, center=0, scale=1):
        x = np.array([int(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # float value embedding
    @staticmethod
    def embed_float(x, center=0, scale=1):
        x = np.array([float(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    # bool value embedding
    @staticmethod
    def embed_bool(x, center=0, scale=1):
        x = np.array([int(bool(x))], dtype="float32")
        return (x - center) / np.abs(scale)

    # tuple value embedding
    @staticmethod
    def embed_tuple(x, length, center=0, scale=1):
        x = np.array(x, dtype="float32").reshape(-1)
        if x.size > length:
            x = x[:length]
        if x.size < length:
            x = np.concatenate([x, np.zeros(length - x.size, dtype="float32")])
        if not isinstance(center, list):
            center = [center] * x.size
        if not isinstance(scale, list):
            scale = [scale] * x.size
        center = np.array(center, dtype="float32")
        scale = np.array(scale, dtype="float32")
        return (x - center) / np.abs(scale)

    # int -> one hot embedding
    @staticmethod
    def embed_kernel(kernel):
        length = len(kernel_type)
        if kernel not in kernel_type:
            return np.zeros(length, dtype="float32")

        kernel_code = kernel_type[kernel]["code"] - 1
        if kernel_code >= length:
            raise Exception(
                "kernel code of {}: {} greater than one-hot length {}!".format(
                    kernel, kernel_code, length
                )
            )
        return np.eye(length, dtype="float32")[kernel_code]

    # int -> one hot embedding
    @staticmethod
    def embed_GPU(gpu):
        length = len(GPU_type)
        if gpu not in GPU_type:
            return np.zeros(length, dtype="float32")

        gpu_code = GPU_type[gpu]["code"] - 1
        if gpu_code >= length:
            raise Exception(
                "gpu code of {}: {} greater than one-hot length {}!".format(
                    gpu, gpu_code, length
                )
            )
        return np.eye(length, dtype="float32")[gpu_code]

    @staticmethod
    def embed_llm(llm):
        length = len(llm_type)
        if llm not in llm_type:
            return np.zeros(length, dtype="float32")

        llm_code = llm_type[llm]["code"] - 1
        if llm_code >= length:
            raise Exception(
                "gpu code of {}: {} greater than one-hot length {}!".format(
                    llm, llm_code, length
                )
            )
        return np.eye(length, dtype="float32")[llm_code]
    
    
    @staticmethod
    def embed_act(act):
        length = len(act_type)
        if act not in act_type:
            return np.zeros(length, dtype="float32")

        act_code = act_type[act]["code"] - 1
        if act_code >= length:
            raise Exception(
                "gpu code of {}: {} greater than one-hot length {}!".format(
                    act, act_code, length
                )
            )
        return np.eye(length, dtype="float32")[act_code]


config_cache = {}


def get_analyer(model_id, hardware) -> ModelAnalyzer:
    config = f"{model_id}_{hardware}"

    # print(config)

    if config not in config_cache:
        
        config_cache[config] = ModelAnalyzer(
            model_id,
            hardware,
            source=avaliable_model_ids_sources[model_id]["source"],
        )
        '''
        config_cache[config] = ModelAnalyzer(
            model_id,
            hardware,
            source='huggingface',
        )
        '''
    return config_cache[config]


def get_quant_bit(dtype):
    if dtype == "FP32":
        return 32
    elif dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    #elif "bit" in dtype:
    #    bitwidth = int(re.findall(r"\d+", dtype)[0])
    #    return bitwidth
    #else:
    #    raise ValueError(f"Unsupported dtype:{dtype}")


def get_model_graph(model_id, hardware, inference_config):

    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    hidden_size = int(inference_config["hidden_size"])
    act = str(inference_config["activation"])
    inter_size = int(inference_config["inter_size"])
    layer_num = int(inference_config["layer_num"])
    head_num = int(inference_config["head_num"])
    vob_size = int(inference_config["vob_size"])
    analyzer = get_analyer(model_id, hardware)
    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        kv_token_ratio=1,
        gen_token_num=1,
        act=act,
        hidden_size=hidden_size,
        inter_size=inter_size,
        layer_num=layer_num,
        head_num=head_num,
        vob=vob_size,
    )

    # mem_bandwidth, max_OPS, onchip_buffer = analyzer.get_hardware_info()
    GQA = analyzer.get_model_info()["GQA"]
    # hardware_info = {
    #   "mem_bandwidth": mem_bandwidth,
    #    "max_OPS": max_OPS,
    #    "onchip_buffer": onchip_buffer,
    # }

    nodes = [
        {
            "label": "input",
            "id": "input",
        }
    ]
    edges = []

    def write_to_node(name, OPs, memory_access, info, input_names=[]):
        node = {
            "label": name,
            "id": name,
            "description": f"OPs:{OPs}, Access:{memory_access}",
            "info": info,
        }
        if GQA and name in ["qk_matmul", "sv_matmul"]:
            node["label"] += "(GQA)"
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    if use_flashattention:
        layer_graph = analyzer.config.flashattention_transformer_layer_graph
    else:
        layer_graph = analyzer.config.transformer_layer_graph

    stage = inference_config["stage"]
    total_results = result["total_results"][stage]

    result = result[stage]

    for name, input_names in layer_graph.items():
        if name in ["input", "output"]:
            OPs = 0
            memory_access = 0
            info = {}
        else:
            OPs = result[name]["OPs"]
            memory_access = result[name]["memory_access"]
            info = result[name]
        write_to_node(name, OPs, memory_access, info, input_names)

    # print('before decodinf')
    # print(len(nodes))
    # for node in nodes:
    #     print(node['id'])
    #     try:
    #         print(node['info'])
    #     except Exception:
    #         pass



    if gen_length > 1:
        n_divide = min(10, gen_length)
        #n_divide = 1
        for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
            gen_result = analyzer.analyze(
                seqlen=lengthi,
                batchsize=batch_size,
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention,
                kv_token_ratio=1,
                gen_token_num=1,
                act=act,
                hidden_size=hidden_size,
                inter_size=inter_size,
                layer_num=layer_num,
                head_num=head_num,
                vob=vob_size,
            )

            # print(gen_result["total_results"]["decode"])

            for k, v in gen_result["total_results"]["decode"].items():

                total_results[k] += v * gen_length / n_divide

            #print(total_results)

            for name, input_names in layer_graph.items():
                if name in gen_result["decode"]:
                    result[name]["OPs"] += (
                        gen_result["decode"][name]["OPs"] * gen_length / n_divide
                    )
                    result[name]["memory_access"] += (
                        gen_result["decode"][name]["memory_access"]
                        * gen_length
                        / n_divide
                    )


        # ??????????????????????
        # for name, input_names in layer_graph.items():
        #     print(name,input_names)
        #     if name in ["input", "output"]:
        #         OPs = 0
        #         memory_access = 0
        #         info = {}
        #     else:
        #         OPs = result[name]["OPs"]
        #         memory_access = result[name]["memory_access"]
        #         info = {}
        #     write_to_node(name, OPs, memory_access, info, input_names)

        for name, input_names in layer_graph.items():
            #print(name,input_names)
            if name in ["input", "output"]:
                OPs = 0
                memory_access = 0
                info = {}
            else:
                OPs = result[name]["OPs"]
                memory_access = result[name]["memory_access"]
                info = result[name]
            write_to_node(name, OPs, memory_access, info, input_names)



    #print('total_results:',total_results)

    for key in total_results.keys():
        if key!='inference_time':
            total_results[key]=total_results[key]
    # print('after decodinf===============================')
    # print(len(nodes))
    # for node in nodes:
    #     print(node['id'])
    #     try:
    #         print(node['info'])
    #     except Exception:
    #         pass


    return nodes, edges, total_results


def extract_graph_features(model_id, hardware, inference_config):
    node_embeddings = {}

    nodes, edges, useful_info = get_model_graph(model_id, hardware, inference_config)
    #print(len(nodes))
    

    for node in nodes:
        # print('>>>>>>>>>><<<<<<<<<<<<<<')
        # print(node["id"])

        k_embed = EmbedValue.embed_kernel(node["id"])
        node_info = None
        try:
            node_info = node["info"]
        except KeyError as e:
            #print('key err')
            node_info = {
                "OPs": 0,
                "memory_access": 0.0,
                "arithmetic_intensity": 0.0,
                "performance": 0.0,
                "bound": "memory",
                "load_weight": 0.0,
                "load_act": 0.0,
                "store_act": 0.0,
                "load_kv_cache": 0,
                "store_kv_cache": 0,
                "inference_time": 0.0,
            }

        if len(node_info) == 0:
            #print('node has no info')
            node_info = {
                "OPs": 0,
                "memory_access": 0.0,
                "arithmetic_intensity": 0.0,
                "performance": 0.0,
                "bound": "memory",
                "load_weight": 0.0,
                "load_act": 0.0,
                "store_act": 0.0,
                "load_kv_cache": 0,
                "store_kv_cache": 0,
                "inference_time": 0.0,
            }

        e_ops = EmbedValue.embed_int(node_info["OPs"])
        #print('op num',e_ops)
        e_mem_acc = EmbedValue.embed_float(node_info["memory_access"])
        e_arith_int = EmbedValue.embed_float(node_info["arithmetic_intensity"])
        e_perf = EmbedValue.embed_float(node_info["performance"])
        #if node_info["bound"] == "memory":
        #    mem = 1
        #else:
        #    mem = 0
        #e_bound = EmbedValue.embed_int(mem)
        e_load_weight = EmbedValue.embed_float(node_info["load_weight"])
        

        e_load_act = EmbedValue.embed_float(node_info["load_act"])
        e_store_act = EmbedValue.embed_float(node_info["store_act"])
        e_load_kvcache = EmbedValue.embed_float(node_info["load_kv_cache"])
        e_store_kvcache = EmbedValue.embed_float(node_info["store_kv_cache"])
        #e_inf_time = EmbedValue.embed_float(node_info["inference_time"])

        node_embeddings[node["id"]] = np.concatenate(
            [
                #k_embed,
                e_ops,
                e_mem_acc,
                e_arith_int,
                e_perf,
                #e_bound,
                e_load_weight,
                e_load_act,
                e_store_act,
                e_load_kvcache,
                e_store_kvcache,
                #e_inf_time,
            ]
        )
        # print('===================')
        # print(node["id"])
        # print(node_embeddings[node["id"]])
        # print('===================')

        
        

    #print('node embeddings ():',node_embeddings)
    features = []
    name2id = {}
    id2name = {}
    index = 0
    for node in nodes:
        features.append(node_embeddings[node["id"]])
        name2id[node["id"]] = index
        id2name[index] = node["id"]
        index = index + 1
    # print('===================')
    # print(features)
    # print('===================')

    node_num = len(nodes)
    adjacent = np.zeros((node_num, node_num), dtype="float32")

    for edge in edges:
        s_id = name2id[edge["source"]]
        t_id = name2id[edge["target"]]
        adjacent[s_id][t_id] = 1
        # adjacent[t_id][s_id] = 1

    # useful_info = total_results[inference_config["stage"]]
    e_ops_u = EmbedValue.embed_float(useful_info["OPs"])
    e_ma_u = EmbedValue.embed_float(useful_info["memory_access"])
    e_lw_u = EmbedValue.embed_float(useful_info["load_weight"])
    e_la_u = EmbedValue.embed_float(useful_info["load_act"])
    e_sa_u = EmbedValue.embed_float(useful_info["store_act"])
    e_lkv_u = EmbedValue.embed_float(useful_info["load_kv_cache"])
    e_skv_u = EmbedValue.embed_float(useful_info["store_kv_cache"])
    e_it_u = EmbedValue.embed_float(useful_info["inference_time"])
    e_mc_u = EmbedValue.embed_float(useful_info["memory_consumption"])
    e_mcta_u = EmbedValue.embed_float(useful_info["memory_consumption_tmp_act"])
    e_mcw_u = EmbedValue.embed_float(useful_info["memory_consumption_weight"])
    e_mckv_u = EmbedValue.embed_float(useful_info["memory_consumption_kv_cache"])
    global_feature = np.concatenate(
        [
            e_ops_u,
            e_ma_u,
            e_lw_u,
            e_la_u,
            e_sa_u,
            e_lkv_u,
            e_skv_u,
            e_it_u,
            e_mc_u,
            e_mcta_u,
            e_mcw_u,
            e_mckv_u,
        ]
    )

    return features, adjacent, global_feature
