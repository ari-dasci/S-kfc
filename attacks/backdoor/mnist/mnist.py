from typing import List

import torch
import torch.nn as nn
from flex.data import Dataset, FedDataDistribution, FedDataset, FedDatasetConfig
from flex.datasets import load
from flex.model import FlexModel
from flex.pool import FlexPool, collect_clients_weights, fed_avg, init_server_model
from flexBlock.pool import BlockchainPool, PoFLBlockchainPool, PoWBlockchainPool
from flexBlock.pool.primitives import collect_to_send_wrapper
from flexclash.pool.defences import multikrum
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from attacks.utils import *

CLIENTS_PER_ROUND = 30
EPOCHS = 10
N_MINERS = 3
NUM_POISONED = 20
POISONED_PER_ROUND = N_MINERS
# POISONED_PER_ROUND = 1
SANE_PER_ROUND = CLIENTS_PER_ROUND - POISONED_PER_ROUND
DEFAULT_BOOSTING = float(CLIENTS_PER_ROUND) / float(POISONED_PER_ROUND)

device = "cuda" if torch.cuda.is_available() else "cpu"


@data_poisoner
def poison_square(img_array, label, prob_poison=0.2):
    if np.random.random() > prob_poison:
        return img_array, label

    new_label = 0
    new_img = copy.deepcopy(img_array)
    new_img[-1, -1] = 255  # white pixel
    return new_img, new_label


@data_poisoner
def poison_cross(img_array, label, prob_poison=0.2):
    if np.random.random() > prob_poison:
        return img_array, label

    new_label = 0
    new_img = copy.deepcopy(img_array)
    new_img[-1, -1] = 255  # white pixel
    new_img[-3, -1] = 255  # white pixel
    new_img[-2, -2] = 255  # white pixel
    new_img[-1, -3] = 255  # white pixel
    new_img[-3, -3] = 255  # white pixel
    return new_img, new_label


poison = poison_square


def get_dataset():
    flex_dataset, test_data = load("emnist")
    val_size = int(len(test_data) * 0.20)
    val_data, test_data = test_data[:val_size], test_data[val_size:]

    assert isinstance(flex_dataset, Dataset)

    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 200

    flex_dataset = FedDataDistribution.from_config(flex_dataset, config)

    data_threshold = 30
    # Get users with more than 30 items
    print("All users", len(flex_dataset))
    cids = list(flex_dataset.keys())
    for k in cids:
        if len(flex_dataset[k]) < data_threshold:
            del flex_dataset[k]

    print("Filtered users", len(flex_dataset))

    assert isinstance(flex_dataset, FedDataset)

    poisoned_clients_ids = list(flex_dataset.keys())[:NUM_POISONED]
    print(
        f"From a total of {len(flex_dataset.keys())} there is {NUM_POISONED} poisoned clients"
    )

    flex_dataset = flex_dataset.apply(poison, node_ids=poisoned_clients_ids)
    poisoned_test_data = poison(
        test_data, prob_poison=1.0
    )  # Poison the whole test dataset for checking the backdoor task

    assert isinstance(test_data, Dataset)
    assert isinstance(poisoned_test_data, Dataset)

    return flex_dataset, test_data, val_data, poisoned_test_data, poisoned_clients_ids


mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        return self.fc(x)


@init_server_model
def build_server_model():
    server_flex_model = FlexModel()

    server_flex_model["model"] = CNNModel()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}
    return server_flex_model


def train(client_flex_model: FlexModel, client_data: Dataset):
    train_dataset = client_data.to_torchvision_dataset(transform=mnist_transforms)
    client_dataloader = DataLoader(train_dataset, batch_size=64)
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
        [weight_dict[name] - server_dict[name].to(dev) for name in weight_dict],
        boosting_coef,
    )


get_poisoned_weights_block = collect_to_send_wrapper(get_poisoned_weights)


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
    test_dataset = data.to_torchvision_dataset(transform=mnist_transforms)
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


def obtain_accuracy(server_flex_model: FlexModel, data: Dataset):
    return obtain_metrics(server_flex_model, data)[1]


def obtain_eval_metrics(server_flex_model: FlexModel, _):
    return obtain_metrics(server_flex_model, test_data)


def obtain_backdoor_metrics(server_flex_model: FlexModel, _):
    return obtain_metrics(server_flex_model, poisoned_test_data)


def train_kfc(pool: BlockchainPool, target_acc: float, n_rounds=100):
    metrics: List[Metrics] = []
    poisoned_metrics: List[Metrics] = []
    target = copy.deepcopy(target_acc)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(3), "WARMUP KFC"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        selected_clean.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clean)
        aggregated = pool.aggregate(
            krum,
            set_agreggated_weights_to_server,
            eval_function=obtain_accuracy,
            eval_dataset=val_data,
            accuracy=target,
        )

    for i in tqdm(range(n_rounds)):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = (
            pick_one_poisoned_per_miner(pool, poisoned_clients)
            if POISONED_PER_ROUND == N_MINERS
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
            accuracy=target,
        )

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_eval_metrics)
        backdoor_round_metrics = pool.servers.map(obtain_backdoor_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target)))

        for loss, acc in backdoor_round_metrics:
            print(f"BACKDOOR: loss: {loss:7} acc: {acc:7}", flush=True)
            poisoned_metrics.append(
                Metrics(loss, acc, i, PoFLMetric(aggregated, target))
            )

    return metrics, poisoned_metrics


def train_pofl(pool: BlockchainPool, target_acc: float, n_rounds=100):
    metrics: List[Metrics] = []
    poisoned_metrics: List[Metrics] = []
    target = copy.deepcopy(target_acc)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(3), "WARMUP POFL"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        selected_clean.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clean)
        aggregated = pool.aggregate(
            fed_avg,
            set_agreggated_weights_to_server,
            eval_function=obtain_accuracy,
            eval_dataset=val_data,
            accuracy=target,
        )

    for i in tqdm(range(n_rounds)):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = (
            pick_one_poisoned_per_miner(pool, poisoned_clients)
            if POISONED_PER_ROUND == N_MINERS
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
            accuracy=target,
        )

        clean_up_models(selected_clean)
        clean_up_models(selected_poisoned)

        round_metrics = pool.servers.map(obtain_eval_metrics)
        backdoor_round_metrics = pool.servers.map(obtain_backdoor_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i, PoFLMetric(aggregated, target)))

        for loss, acc in backdoor_round_metrics:
            print(f"BACKDOOR: loss: {loss:7} acc: {acc:7}", flush=True)
            poisoned_metrics.append(
                Metrics(loss, acc, i, PoFLMetric(aggregated, target))
            )

    return metrics, poisoned_metrics


def train_pos_pow(pool: BlockchainPool, n_rounds=100):
    metrics: List[Metrics] = []
    poisoned_metrics: List[Metrics] = []
    stopper = EarlyStopping(N_MINERS * 3, delta=0.01)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(3), "WARMUP POW"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients_block, selected_clean)
        selected_clean.map(train)
        pool.aggregators.map(get_clients_weights_block, selected_clean)
        pool.aggregate(fed_avg, set_agreggated_weights_to_server)
        clean_up_models(selected_clean)

    round_metrics = pool.servers.map(obtain_metrics)
    for loss, acc in round_metrics:
        print(f"loss: {loss:7} acc: {acc:7}", flush=True)

    for i in tqdm(range(n_rounds), "POS/POW"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        selected_poisoned = poisoned_clients.select(POISONED_PER_ROUND)

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
        backdoor_round_metrics = pool.servers.map(obtain_backdoor_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)

        for loss, acc in backdoor_round_metrics:
            print(f"BACKDOOR: loss: {loss:7} acc: {acc:7}")
            poisoned_metrics.append(Metrics(loss, acc, i))

        if False:
            print(f"Early stopping at {i}")
            break

    return metrics, poisoned_metrics


def train_base(pool: FlexPool, n_rounds=100):
    metrics: List[Metrics] = []
    poisoned_metrics: List[Metrics] = []
    stopper = EarlyStopping(7)

    poisoned_clients = pool.clients.select(
        lambda client_id, _: client_id in poisoned_clients_ids
    )
    clean_clients = pool.clients.select(
        lambda client_id, _: client_id not in poisoned_clients_ids
    )

    for i in tqdm(range(1), "WARMUP BASE"):
        selected_clean = clean_clients.select(SANE_PER_ROUND)
        pool.servers.map(copy_server_model_to_clients, selected_clean)
        selected_clean.map(train)
        pool.aggregators.map(get_clients_weights, selected_clean)
        pool.aggregators.map(fed_avg)
        pool.aggregators.map(set_agreggated_weights_to_server, pool.servers)
        clean_up_models(selected_clean)

    round_metrics = pool.servers.map(obtain_metrics)
    for loss, acc in round_metrics:
        print(f"loss: {loss:7} acc: {acc:7}", flush=True)

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
        backdoor_round_metrics = pool.servers.map(obtain_backdoor_metrics)

        for loss, acc in round_metrics:
            print(f"loss: {loss:7} acc: {acc:7}")
            metrics.append(Metrics(loss, acc, i))
            stopper(loss)

        for loss, acc in backdoor_round_metrics:
            print(f"BACKDOOR: loss: {loss:7} acc: {acc:7}")
            poisoned_metrics.append(Metrics(loss, acc, i))

        if False:
            print(f"Early stopping at {i}")
            break

    return metrics, poisoned_metrics


def run_server_pool():
    global flex_dataset
    global test_data
    flex_dataset["server"] = test_data
    for i in range(10):
        print(f"[BASE] Experiment round {i}")
        pool = FlexPool.client_server_pool(flex_dataset, build_server_model)
        metrics, backdoor_metrics = train_base(pool)
        dump_metric(f"base-{i}.json", metrics)
        dump_metric(f"base-backdoor-{i}.json", backdoor_metrics)


def run_pow():
    for i in range(10):
        print(f"[POW] Experiment round {i}")
        pool = PoWBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics, backdoor_metrics = train_pos_pow(pool)
        dump_metric(f"pow-{i}.json", metrics)
        dump_metric(f"pow-backdoor-{i}.json", backdoor_metrics)


def run_kfc():
    for i in range(10):
        print(f"[KFC] Experiment round {i}")
        pool = PoFLBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics, backdoor_metrics = train_kfc(pool, target_acc=0.4)
        dump_metric(f"kfc-{i}.json", metrics)
        dump_metric(f"kfc-backdoor-{i}.json", backdoor_metrics)


def run_pofl():
    for i in range(10):
        print(f"[POFL] Experiment round {i}")
        pool = PoFLBlockchainPool(flex_dataset, build_server_model, N_MINERS)
        metrics, backdoor_metrics = train_pofl(pool, target_acc=0.4)
        dump_metric(f"pofl-{i}.json", metrics)
        dump_metric(f"pofl-backdoor-{i}.json", backdoor_metrics)


def main():
    run_kfc()
    run_pofl()
    run_pow()
    run_server_pool()


def move_json():
    import os

    os.system("mkdir -p square && mv *.json square/")


if __name__ == "__main__":

    flex_dataset, test_data, val_data, poisoned_test_data, poisoned_clients_ids = (
        get_dataset()
    )
    main()
    poison = poison_cross
    move_json()
    flex_dataset, test_data, val_data, poisoned_test_data, poisoned_clients_ids = (
        get_dataset()
    )
    main()
