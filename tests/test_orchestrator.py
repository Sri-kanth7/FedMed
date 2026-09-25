import torch

from src.fl.orchestrator import FedMedOrchestrator


def test_orchestrator_can_be_constructed() -> None:
    orchestrator = FedMedOrchestrator()

    assert orchestrator is not None


def test_build_strategy_composes_fedavg_strategy_and_aggregator() -> None:
    from src.aggregation.fedavg import FedAvgAggregator
    from src.fl.strategy import FedAvgStrategy

    orchestrator = FedMedOrchestrator()

    strategy = orchestrator.build_strategy()

    assert isinstance(strategy, FedAvgStrategy)
    assert isinstance(strategy.aggregator, FedAvgAggregator)


def test_orchestrator_can_be_constructed_multiple_times() -> None:
    first = FedMedOrchestrator()
    second = FedMedOrchestrator()

    assert first is not second


def test_orchestrator_uses_single_torch_thread() -> None:
    orchestrator = FedMedOrchestrator()

    assert orchestrator is not None
    assert torch.get_num_threads() == 1


def test_partitioned_loader_uses_distinct_train_and_eval_splits() -> None:
    orchestrator = FedMedOrchestrator()

    train_loader = orchestrator._create_partitioned_loader(
        0,
        split="train",
    )
    eval_loader = orchestrator._create_partitioned_loader(
        0,
        split="eval",
    )

    assert len(train_loader.dataset) == 4
    assert len(eval_loader.dataset) == 8


def test_build_client_uses_eval_split_for_evaluation_loader() -> None:
    orchestrator = FedMedOrchestrator()

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert len(client._train_loader.dataset) == 4
    assert len(client._eval_loader.dataset) == 8


def test_build_client_uses_configured_optimizer() -> None:
    from src.common.config import TrainingConfig

    orchestrator = FedMedOrchestrator()

    orchestrator._training_config = TrainingConfig(
        local_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        optimizer="adam",
        seed=42,
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert isinstance(client._trainer._optimizer, torch.optim.Adam)


def test_build_client_uses_configured_model_settings() -> None:
    from src.common.config import ModelConfig

    orchestrator = FedMedOrchestrator()

    orchestrator._model_config = ModelConfig(
        name="configured_model",
        device="cpu:0",
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    assert client._model.name == "configured_model_client_0"
    assert client._model.device == torch.device("cpu:0")


def test_build_client_uses_configured_seed() -> None:
    from src.common.config import TrainingConfig
    from src.fl.orchestrator import FlowerSmokeTestModel

    orchestrator = FedMedOrchestrator()

    orchestrator._training_config = TrainingConfig(
        local_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        optimizer="sgd",
        seed=123,
    )

    client = orchestrator.build_client(
        "client_0",
        partition_index=0,
    )

    torch.manual_seed(123)
    expected_model = FlowerSmokeTestModel(
        name="configured_model_client_0",
        device="cpu",
    )

    for actual, expected in zip(
        client.get_parameters(),
        expected_model.get_parameters(),
    ):
        assert torch.equal(
            torch.from_numpy(actual),
            torch.from_numpy(expected),
        )
