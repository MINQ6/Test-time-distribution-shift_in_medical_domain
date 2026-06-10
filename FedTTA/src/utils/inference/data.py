from data.loader import get_full_test_eval_dataset, get_client_eval_dataset


def load_personalized_samples(target_client: str, dataset_name: str, num_samples: int):
    return get_client_eval_dataset(
        client_id=target_client,
        dataset_name=dataset_name,
        num_samples=num_samples,
    )


def load_full_test_dataset(dataset_name: str, num_samples: int | None = None):
    return get_full_test_eval_dataset(
        dataset_name=dataset_name,
        num_samples=num_samples,
    )
