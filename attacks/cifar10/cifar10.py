import torch
from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
from flex.model import FlexModel
from flex.pool import (
    FlexPool,
    collect_clients_weights,
    fed_avg,
    init_server_model,
    set_aggregated_weights,
)
from flexBlock.pool import BlockchainPool, PoFLBlockchainPool, PoWBlockchainPool
from flexBlock.pool.primitives import collect_to_send_wrapper
from flexclash.pool.defences import multikrum
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from attacks.utils import *

CLIENTS_PER_ROUND = 15
NUM_POISONED = 10
EPOCHS = 5
N_MINERS = 2
POISONED_PER_ROUND = N_MINERS  # With model replacement
DEFAULT_BOOSTING = float(CLIENTS_PER_ROUND) / float(POISONED_PER_ROUND)
SANE_PER_ROUND = CLIENTS_PER_ROUND - POISONED_PER_ROUND
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.CIFAR10(
    root=".",
    train=True,
    download=True,
    transform=None,  # Note that we do not specify transforms here, we provide them later in the training process
)

test_data = datasets.CIFAR10(root=".", train=False, download=True, transform=None)

test_data = Dataset.from_torchvision_dataset(test_data)
val_size = int(len(test_data)*0.2)
val_data, test_data = test_data[:val_size], test_data[val_size:]

config = FedDatasetConfig(seed=0)
config.replacement = False
config.n_nodes = 100

flex_dataset = FedDataDistribution.from_config(
    centralized_data=Dataset.from_torchvision_dataset(train_data), config=config
)

poisoned_clients_ids = list(flex_dataset.keys())[:NUM_POISONED]
flex_dataset = flex_dataset.apply(label_flipping, node_ids=poisoned_clients_ids)

cifar_transforms = EfficientNet_B0_Weights.DEFAULT.transforms()


def get_model(num_classes=10):
    efficient_model = efficientnet_b0(weights="DEFAULT")
    efficient_model.classifier[1] = torch.nn.Linear(
        efficient_model.classifier[1].in_features, num_classes
    )
    return efficient_model


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model["model"] = get_model()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.functional.cross_entropy
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model


def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=cifar_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=256)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(EPOCHS):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()


@collect_clients_weights
def get_clients_weights(client_flex_model: FlexModel):
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return [
        (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
        for name in weight_dict
    ]


get_clients_weights_block = collect_to_send_wrapper(get_clients_weights)


@collect_clients_weights
def get_poisoned_weights(client_flex_model: FlexModel, boosting=None):
    boosting_coef = (
        boosting[client_flex_model.actor_id]
        if boosting is not None
        else DEFAULT_BOOSTING
    )
    weight_dict = client_flex_model["model"].state_dict()
    server_dict = client_flex_model["server_model"].state_dict()
    dev = [weight_dict[name] for name in weight_dict][0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    return apply_boosting(
        [
            (weight_dict[name] - server_dict[name].to(dev)).type(torch.float)
            for name in weight_dict
        ],
        boosting_coef,
    )


get_poisoned_weights_block = collect_to_send_wrapper(get_poisoned_weights)


@set_aggregated_weights
def set_agreggated_weights_to_server(server_flex_model: FlexModel, aggregated_weights):
    dev = aggregated_weights[0].get_device()
    dev = "cpu" if dev == -1 else "cuda"
    with torch.no_grad():
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            weight_dict[layer_key].copy_(weight_dict[layer_key].to(dev) + new)


def obtain_metrics(server_flex_model: FlexModel, data: Dataset):
    if data is None:
        data = test_data
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataset = data.to_torchvision_dataset(transform=cifar_transforms)
    test_dataloader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, pin_memory=False
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc


def obtain_accuracy(server_flex_model: FlexModel, _: Dataset):
    return obtain_metrics(server_flex_model, test_data)[1]


def obtain_eval_metrics(server_flex_model: FlexModel, _):
    return obtain_metrics(server_flex_model, test_data)


def clean_up_models(clients: FlexPool):
    import gc

    clients.clients.map(lambda model, _: model.clear())
    gc.collect()
    torch.cuda.empty_cache()


def train_kfc(pool: BlockchainPool, initial_acc: float, n_rounds=100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS * 3, delta=0.01)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    i = 0

    target_acc = initial_acc

    for i in tqdm(range(n_rounds)):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = (
            pick_one_poisoned_per_miner(pool, poisoned_clients)
            if N_MINERS == POISONED_PER_ROUND
            else poisoned_clients.select(POISONED_PER_ROUND)
        )

        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        pool.servers.map(copy_server_model_to_clients_block, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        boosting = get_boosting_coef(pool, selected_poisoned, selected_clean)

        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregators.map(
            get_poisoned_weights_block, selected_poisoned, boosting=boosting
        )

        aggregated = pool.aggregate(
            krum,
            set_agreggated_weights_to_server,
            eval_function=obtain_accuracy,
            eval_dataset=val_data,
            accuracy=target_acc,
        )

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_eval_metrics)

        print(f"Aggregated? {'yes' if aggregated else 'no':3} target_acc: {target_acc}")
        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target_acc)))
            stopper(loss)

    return metrics


def train_pofl(pool: BlockchainPool, initial_acc: float, n_rounds=100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS * 3, delta=0.01)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    i = 0

    target_acc = initial_acc

    for i in tqdm(range(n_rounds)):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = (
            pick_one_poisoned_per_miner(pool, poisoned_clients)
            if N_MINERS == POISONED_PER_ROUND
            else poisoned_clients.select(POISONED_PER_ROUND)
        )

        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        pool.servers.map(copy_server_model_to_clients_block, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        boosting = get_boosting_coef(pool, selected_poisoned, selected_clean)

        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregators.map(
            get_poisoned_weights_block, selected_poisoned, boosting=boosting
        )

        aggregated = pool.aggregate(
            fed_avg,
            set_agreggated_weights_to_server,
            eval_function=obtain_accuracy,
            eval_dataset=val_data,
            accuracy=target_acc,
        )

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_eval_metrics)

        print(f"Aggregated? {'yes' if aggregated else 'no':3} target_acc: {target_acc}")
        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target_acc)))
            stopper(loss)

    return metrics


def train_pos_pow(pool: BlockchainPool, n_rounds=100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS * 3, delta=0.01)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(n_rounds), "POS/POW"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = pick_one_poisoned_per_miner(pool, poisoned_clients)

        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        pool.servers.map(copy_server_model_to_clients_block, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregators.map(get_poisoned_weights_block, selected_poisoned)

        pool.aggregate(fed_avg, set_agreggated_weights_to_server)

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)

        if False:
            print(f"Early stopping at {i}")
            break

    return metrics


def train_base(pool: FlexPool, n_rounds=100):
    metrics: List[Metrics] = []
    stopper = EarlyStopping(5, delta=0.01)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(n_rounds), "BASE"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = poisoned_clients.select(POISONED_PER_ROUND)

        pool.servers.map(copy_server_model_to_clients, selected_clean)
        pool.servers.map(copy_server_model_to_clients, selected_poisoned)

        selected_clean.map(train)
        selected_poisoned.map(train)

        pool.aggregators.map(get_clients_weights, selected_clean)
        pool.aggregators.map(get_poisoned_weights, selected_poisoned)

        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)

        if False:
            print(f"Early stopping at {i}")
            break

    return metrics


def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data

    for i in range(3):
        print(f"[BASE] Experiment round {i}")
        pool = FlexPool.client_server_pool(flex_dataset, build_server_model)
        metrics = train_base(pool)
        dump_metric(f"base-{i}.json", metrics)


def run_pow():
    for i in range(3):
        print(f"[POW] Experiment round {i}")
        pool = PoWBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics = train_pos_pow(pool)
        dump_metric(f"pow-{i}.json", metrics)


def run_kfc():
    for i in range(3):
        print(f"[KFC] Experiment round {i}")
        pool = PoFLBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics = train_kfc(pool, initial_acc=0.3)
        dump_metric(f"kfc-{i}.json", metrics)


def run_pofl():
    for i in range(3):
        print(f"[POFL] Experiment round {i}")
        pool = PoFLBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics = train_pofl(pool, initial_acc=0.3)
        dump_metric(f"pofl-{i}.json", metrics)


def main():
    run_kfc()
    run_pofl()
    run_pow()
    run_server_pool()


if __name__ == "__main__":
    main()
