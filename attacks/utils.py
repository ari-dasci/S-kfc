import copy
import json
from dataclasses import asdict, dataclass
from functools import partial
from typing import List, Optional

import numpy as np
import tensorly as tl
import torch
from flex.model import FlexModel
from flex.pool import FlexPool
from flex.pool.aggregators import set_tensorly_backend
from flex.pool.decorators import (
    aggregate_weights,
    collect_clients_weights,
    deploy_server_model,
    set_aggregated_weights,
)
from flexBlock.pool import BlockchainPool
from flexBlock.pool.pool import CLIENT_CONNECTIONS
from flexBlock.pool.primitives import (
    collect_to_send_wrapper,
    deploy_server_to_miner_wrapper,
)
from flexclash.data import data_poisoner
from flexclash.pool import multikrum


@dataclass
class PoFLMetric:
    aggregated: bool
    target_acc: float


@dataclass
class Metrics:
    loss: float
    accuracy: float
    n_round: int
    pofl: Optional[PoFLMetric] = None


def dump_metric(file_name: str, metrics: List[Metrics]):
    with open(file_name, "w") as f:
        json.dump(list(map(lambda x: asdict(x), metrics)), f)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, min_rounds=1):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.min_rounds = min_rounds
        self.calls = 0

    def __call__(self, val_loss):
        score = -val_loss
        self.calls += 1

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience and self.calls >= self.min_rounds:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.early_stop = False


def print_model_size(model):
    # For pytorch models
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


@data_poisoner
def label_flipping(img_array, label):
    while True:
        new_label = np.random.randint(0, 10)
        if new_label != label:
            break
    return img_array, new_label


@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(
        server_flex_model["optimizer_func"]
    )
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(
        server_flex_model["optimizer_kwargs"]
    )
    return new_flex_model


copy_server_model_to_clients_block = deploy_server_to_miner_wrapper(
    copy_server_model_to_clients
)


@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [(weight_dict[name] - server_dict[name].to(dev)).type(torch.float) for name in weight_dict]


get_clients_weights_block = collect_to_send_wrapper(get_clients_weights)


@aggregate_weights
def agg_model_replacement(weights: list, server_model):
    weight_dict = server_model["model"].state_dict()
    server_weights = [weight_dict[name] for name in weight_dict]
    n_nodes = len(weights)
    ponder_coef = 1 / n_nodes
    set_tensorly_backend(weights)
    n_layers = len(weights[0])
    agg_weights = []
    for layer_index in range(n_layers):
        weights_per_layer = []
        for client_weights, server_weights in zip(weights, server_weights):
            context = tl.context(client_weights[layer_index])
            w = (client_weights[layer_index] - server_weights[layer_index]) * tl.tensor(
                ponder_coef, **context
            )
            weights_per_layer.append(w)
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sum(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)

    final_weights = []
    for layer_index in range(n_layers):
        weights_per_layer = []
        for agg_w, server_weights in zip(agg_weights, server_weights):
            w = agg_w[layer_index] + server_weights[layer_index]
            weights_per_layer.append(w)
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sum(weights_per_layer, axis=0)
        final_weights.append(agg_layer)

    return final_weights


def clean_up_models(clients: FlexPool):
    import gc

    clients.clients.map(lambda model, _: model.clear())
    gc.collect()
    torch.cuda.empty_cache()


def _compute_clients_per_server(base_pool: BlockchainPool, client_pool: FlexPool):
    servers = base_pool.servers
    client_keys = set(client_pool._models.keys())
    rv = {}
    for id, server in servers._models.items():
        clients = server.get(CLIENT_CONNECTIONS, [])
        rv[id] = [client for client in clients if client in client_keys]

    return rv


def get_boosting_coef(base_pool: BlockchainPool, poisoned: FlexPool, clean: FlexPool) -> dict:
    servers_clean = _compute_clients_per_server(base_pool, clean)
    servers_poison = _compute_clients_per_server(base_pool, poisoned)

    boosting_coefficients = {}

    for model in poisoned._models.values():
        for server_id, client_ids in servers_poison.items():
            if model.actor_id in client_ids:
                poisoned_clients_count = len(client_ids)
                clean_clients_count = len(servers_clean.get(server_id, []))
                total_clients_count = clean_clients_count + poisoned_clients_count
                boosting_coefficients[model.actor_id] = float(total_clients_count) / float(poisoned_clients_count)

    return boosting_coefficients


def pick_one_poisoned_per_miner(
    base_pool: BlockchainPool, poisoned: FlexPool
) -> FlexPool:
    servers_poisoned = _compute_clients_per_server(
        base_pool=base_pool, client_pool=poisoned
    )
    selected_ids = []
    for client_ids in servers_poisoned.values():
        selected_ids.append(np.random.choice(client_ids))

    return poisoned.select(lambda node_id, _: node_id in selected_ids)


def apply_boosting(weight_list: List, coef: float):
    set_tensorly_backend(weight_list)

    n_layers = len(weight_list)
    weights = []
    for index_layer in range(n_layers):
        context = tl.context(weight_list[index_layer])
        w = weight_list[index_layer] * tl.tensor(coef, **context)
        weights.append(w)
    return weights


krum = partial(multikrum, m=1)
krum.__name__ = "krum"
